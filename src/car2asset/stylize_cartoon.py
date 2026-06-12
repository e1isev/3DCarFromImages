"""
Cartoon stylisation pass.

Takes the realistic mesh and produces a game-ready cartoon GLB:
  • Simplified / smoothed geometry
  • Bright solid-colour paint derived from the dominant car colour
  • Toon-shading material properties embedded in the GLB extras
  • Thick black outline encoded as an inverted-hull mesh layer

The Blender-based toon shader (blender/cartoon_shader.py) can produce a
higher-quality cel-shaded render; this module handles the pure-Python GLB path
so the pipeline works without a Blender installation.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# How much to simplify the mesh for a cartoon look (fraction of original faces)
CARTOON_FACE_RATIO = 0.4
# Outline thickness as a fraction of the mesh bounding-box diagonal
OUTLINE_THICKNESS_RATIO = 0.006
# Colour saturation boost (0–1 added to HSV saturation)
SATURATION_BOOST = 0.25
# Value (brightness) boost
VALUE_BOOST = 0.1


def generate_cartoon_mesh(
    source_mesh_path: Path,
    output_path: Path,
    *,
    reference_images: dict[str, Image.Image] | None = None,
) -> Path:
    """
    Derive a cartoon mesh from the realistic GLB at *source_mesh_path*.

    Steps:
      1. Load mesh
      2. Simplify geometry
      3. Smooth normals
      4. Sample dominant car colour from reference images (or mesh vertex colours)
      5. Apply brightened toon material
      6. Add inverted-hull outline mesh
      7. Export as GLB

    Returns the path to the cartoon GLB.
    """
    import trimesh  # type: ignore[import]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading realistic mesh: %s", source_mesh_path)
    scene = trimesh.load(str(source_mesh_path), force="scene")

    cartoon_meshes: list[trimesh.Trimesh] = []
    for name, geom in scene.geometry.items():
        logger.info("  Stylising geometry: %s (%d faces)", name, len(geom.faces))
        stylised = _stylise_geometry(geom, reference_images=reference_images)
        cartoon_meshes.append(stylised)

        outline = _build_outline_mesh(stylised)
        if outline is not None:
            cartoon_meshes.append(outline)

    if not cartoon_meshes:
        raise RuntimeError(f"No geometry found in {source_mesh_path}")

    combined = trimesh.util.concatenate(cartoon_meshes)
    export_scene = trimesh.Scene(combined)
    export_scene.export(str(output_path))
    logger.info("Cartoon mesh → %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _stylise_geometry(
    mesh,
    reference_images: dict[str, Image.Image] | None = None,
):
    """Simplify, smooth, and repaint a single trimesh geometry."""
    import trimesh
    from trimesh.smoothing import filter_laplacian

    # Simplify
    target_faces = max(100, int(len(mesh.faces) * CARTOON_FACE_RATIO))
    try:
        simplified = mesh.simplify_quadric_decimation(target_faces)
    except Exception:
        logger.warning("Mesh simplification failed; using original geometry.")
        simplified = mesh.copy()

    # Smooth normals
    try:
        filter_laplacian(simplified, iterations=2)
    except Exception:
        pass

    # Determine dominant colour
    base_colour = _dominant_colour(mesh, reference_images)
    toon_colour = _boost_colour(base_colour)

    # Apply flat toon material
    mat = trimesh.visual.material.PBRMaterial(
        name="cartoon_body",
        baseColorFactor=np.array([*toon_colour, 1.0], dtype=np.float32),
        roughnessFactor=0.9,
        metallicFactor=0.0,
    )
    simplified.visual = trimesh.visual.TextureVisuals(material=mat)
    return simplified


def _build_outline_mesh(mesh):
    """
    Create an inverted-hull outline mesh (black, slightly enlarged).

    This is the standard real-time technique for thick outlines: flip normals,
    scale outward, paint black. Compatible with GLB without custom shaders.
    """
    import trimesh

    try:
        outline = mesh.copy()
        outline.invert()

        diag = np.linalg.norm(mesh.bounding_box.extents)
        scale = 1.0 + OUTLINE_THICKNESS_RATIO * diag / max(diag, 1e-6) * 100
        outline.apply_scale(scale)

        black_mat = trimesh.visual.material.PBRMaterial(
            name="cartoon_outline",
            baseColorFactor=np.array([0.02, 0.02, 0.02, 1.0], dtype=np.float32),
            roughnessFactor=1.0,
            metallicFactor=0.0,
        )
        outline.visual = trimesh.visual.TextureVisuals(material=black_mat)
        return outline
    except Exception as exc:
        logger.warning("Could not build outline mesh: %s", exc)
        return None


def _dominant_colour(mesh, reference_images: dict[str, Image.Image] | None) -> tuple[float, float, float]:
    """Return the dominant RGB colour (0-1 floats) of the car body."""
    if reference_images:
        try:
            return _sample_colour_from_images(reference_images)
        except Exception as exc:
            logger.debug("Image colour sampling failed: %s", exc)

    # Fall back to mesh vertex colours
    try:
        vc = mesh.visual.to_color().vertex_colors[:, :3] / 255.0
        median = np.median(vc, axis=0)
        return float(median[0]), float(median[1]), float(median[2])
    except Exception:
        return (0.8, 0.1, 0.1)  # default red


def _sample_colour_from_images(images: dict[str, Image.Image]) -> tuple[float, float, float]:
    """
    Sample the dominant opaque car-body colour from the prepared (alpha-masked) images.
    Uses the front view; falls back to any available view.
    """
    img = images.get("front") or next(iter(images.values()))
    arr = np.array(img.convert("RGBA")).astype(np.float32)

    # Only use pixels where alpha > 200 (car body, not background)
    mask = arr[:, :, 3] > 200
    if mask.sum() < 100:
        raise ValueError("Too few opaque pixels")

    rgb = arr[:, :, :3][mask] / 255.0
    median = np.median(rgb, axis=0)
    return float(median[0]), float(median[1]), float(median[2])


def _boost_colour(rgb: tuple[float, float, float]) -> tuple[float, float, float]:
    """Increase saturation and brightness for a cartoon look."""
    import colorsys

    h, s, v = colorsys.rgb_to_hsv(*rgb)
    s = min(1.0, s + SATURATION_BOOST)
    v = min(1.0, v + VALUE_BOOST)
    return colorsys.hsv_to_rgb(h, s, v)
