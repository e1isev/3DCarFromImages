"""
3D mesh reconstruction from multi-view car images.

Supported backends (selected via --quality / config):
  fast   → TripoSR  (single-image, ~6 GB VRAM, fast)
  high   → Hunyuan3D-2mv (multi-view, ~10 GB VRAM, better consistency)
  ultra  → Hunyuan3D-2mv + Hunyuan3D-Paint PBR texture (~29 GB VRAM)
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

from car2asset.config import Backend, Quality, settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def generate_realistic_mesh(
    images: dict[str, Image.Image],
    output_path: Path,
    quality: Quality = "high",
) -> Path:
    """
    Generate a realistic 3D mesh from the four prepared car images.

    Returns the path to the output .glb file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    backend: Backend = settings.backend_for_quality if quality == settings.quality else _quality_to_backend(quality)

    logger.info("Reconstruction backend: %s (quality=%s)", backend, quality)

    if backend == "triposr":
        return _run_triposr(images, output_path)
    elif backend == "stable-fast-3d":
        return _run_stable_fast_3d(images, output_path)
    elif backend == "hunyuan3d":
        if quality == "ultra":
            return _run_hunyuan3d(images, output_path, with_pbr_texture=True)
        return _run_hunyuan3d(images, output_path, with_pbr_texture=False)
    else:
        raise ValueError(f"Unknown backend: {backend}")


# ---------------------------------------------------------------------------
# Backend: TripoSR
# ---------------------------------------------------------------------------


def _run_triposr(images: dict[str, Image.Image], output_path: Path) -> Path:
    """Single-image TripoSR reconstruction using the front view."""
    try:
        import tsr  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError(
            "TripoSR is not installed.\n"
            "Install with: pip install git+https://github.com/vast-ai-research/TripoSR.git"
        ) from exc

    import torch

    logger.info("Running TripoSR on front view …")
    front_image = images["front"].convert("RGB")

    device = settings.device
    model = tsr.TSR.from_pretrained("stabilityai/TripoSR", config_name="config.yaml", weight_name="model.ckpt")
    model.renderer.set_chunk_size(131072)
    model.to(device)

    with torch.no_grad():
        scene_codes = model([front_image], device=device)
        meshes = model.extract_mesh(scene_codes, resolution=256)

    mesh = meshes[0]
    _export_trimesh_to_glb(mesh, output_path)
    logger.info("TripoSR → %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Backend: Stable Fast 3D
# ---------------------------------------------------------------------------


def _run_stable_fast_3d(images: dict[str, Image.Image], output_path: Path) -> Path:
    """Stable Fast 3D reconstruction using the front view."""
    try:
        import sf3d  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError(
            "Stable Fast 3D is not installed.\n"
            "Install with: pip install git+https://github.com/Stability-AI/stable-fast-3d.git"
        ) from exc

    import torch

    logger.info("Running Stable Fast 3D on front view …")
    front_image = images["front"].convert("RGBA")

    model = sf3d.StableFast3D.from_pretrained("stabilityai/stable-fast-3d")
    model.to(settings.device)

    with torch.no_grad():
        mesh, _, _ = model.run_image(front_image, bake_resolution=1024)

    _export_trimesh_to_glb(mesh, output_path)
    logger.info("Stable Fast 3D → %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Backend: Hunyuan3D
# ---------------------------------------------------------------------------


def _run_hunyuan3d(
    images: dict[str, Image.Image],
    output_path: Path,
    *,
    with_pbr_texture: bool = False,
) -> Path:
    """
    Multi-view Hunyuan3D-2mv reconstruction.

    with_pbr_texture=True runs the additional Hunyuan3D-Paint pass for
    PBR textures (~29 GB VRAM total). Set quality='ultra' to enable.
    """
    try:
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError(
            "Hunyuan3D is not installed.\n"
            "Install with: pip install git+https://github.com/Tencent/Hunyuan3D-2.git\n"
            "System requirements: CUDA GPU with ≥10 GB VRAM for shape, ≥21 GB for texture."
        ) from exc

    import torch

    model_id = settings.hunyuan3d_shape_model_path
    logger.info("Loading Hunyuan3D shape pipeline from %s …", model_id)

    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        model_id,
        subfolder=settings.hunyuan3d_shape_subfolder,
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    pipeline.to(settings.device)

    # Build ordered PIL image list: front / rear / left / right
    view_order = ["front", "right", "rear", "left"]
    mv_images = [images[v].convert("RGBA") for v in view_order if v in images]

    logger.info("Running Hunyuan3D shape generation (%d views) …", len(mv_images))
    with torch.no_grad():
        result = pipeline(image=mv_images[0], extra_images=mv_images[1:])

    mesh = result.meshes[0]

    if with_pbr_texture:
        mesh = _apply_hunyuan3d_pbr_texture(mesh, mv_images, output_path.parent)

    _export_trimesh_to_glb(mesh, output_path)
    logger.info("Hunyuan3D → %s", output_path)
    return output_path


def _apply_hunyuan3d_pbr_texture(mesh, mv_images: list[Image.Image], output_dir: Path):
    """Run Hunyuan3D-Paint PBR texture generation on a reconstructed mesh."""
    try:
        from hy3dgen.texgen import Hunyuan3DPaintPipeline  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError(
            "Hunyuan3D texture pipeline not found. Ensure Hunyuan3D-2 is installed."
        ) from exc

    import torch

    logger.info("Running Hunyuan3D PBR texture generation (requires ~21 GB VRAM) …")
    tex_pipeline = Hunyuan3DPaintPipeline.from_pretrained(
        settings.hunyuan3d_texture_model_path, torch_dtype=torch.float16
    )
    tex_pipeline.to(settings.device)

    result = tex_pipeline(mesh=mesh, image=mv_images[0])

    # Save individual PBR maps
    tex_dir = output_dir / "textures"
    tex_dir.mkdir(exist_ok=True)
    for map_name in ("basecolor", "roughness", "metallic", "normal"):
        tex_map = getattr(result, f"{map_name}_map", None)
        if tex_map is not None:
            tex_map.save(tex_dir / f"{map_name}.png")
            logger.info("  Saved %s.png", map_name)

    return result.mesh


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _export_trimesh_to_glb(mesh, output_path: Path) -> None:
    """Export a trimesh Mesh/Scene to GLB format."""
    import trimesh  # type: ignore[import]

    if isinstance(mesh, trimesh.Scene):
        mesh.export(str(output_path))
    else:
        scene = trimesh.Scene(mesh)
        scene.export(str(output_path))


def _quality_to_backend(quality: Quality) -> Backend:
    return {"fast": "triposr", "high": "hunyuan3d", "ultra": "hunyuan3d"}[quality]
