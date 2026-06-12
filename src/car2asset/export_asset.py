"""
Asset export utilities.

Handles:
  • GLB export (trimesh-native, no extra deps)
  • FBX export via Blender headless subprocess
  • Output directory structure
"""

from __future__ import annotations

import json
import logging
import subprocess
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

BLENDER_FBX_SCRIPT = Path(__file__).parent.parent.parent / "blender" / "export_fbx.py"


def export_glb(mesh_path: Path, output_path: Path) -> Path:
    """
    Copy (or re-export) a mesh to a canonical .glb path.

    If *mesh_path* is already a .glb, it is simply copied; otherwise trimesh
    loads and re-exports it to ensure a clean, minimal file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if mesh_path.suffix.lower() == ".glb" and mesh_path != output_path:
        shutil.copy2(mesh_path, output_path)
        logger.info("GLB copied → %s", output_path)
        return output_path

    import trimesh  # type: ignore[import]

    scene = trimesh.load(str(mesh_path), force="scene")
    scene.export(str(output_path))
    logger.info("GLB exported → %s", output_path)
    return output_path


def export_fbx(glb_path: Path, output_path: Path, blender_path: Path | None = None) -> Path:
    """
    Convert a GLB to FBX using Blender's headless Python API.

    Requires Blender ≥3.6 installed and the BLENDER_PATH env var / config set.
    """
    from car2asset.config import settings

    blender = blender_path or settings.blender_path
    if not blender.exists():
        raise RuntimeError(
            f"Blender not found at {blender}. "
            "Install Blender and set BLENDER_PATH in .env, or skip FBX export."
        )

    if not BLENDER_FBX_SCRIPT.exists():
        raise RuntimeError(f"Blender FBX export script not found: {BLENDER_FBX_SCRIPT}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(blender),
        "--background",
        "--python", str(BLENDER_FBX_SCRIPT),
        "--",
        str(glb_path),
        str(output_path),
    ]
    logger.info("Running Blender FBX export: %s → %s", glb_path.name, output_path.name)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    if result.returncode != 0:
        logger.error("Blender stderr:\n%s", result.stderr)
        raise RuntimeError(f"Blender FBX export failed (exit {result.returncode})")

    logger.info("FBX exported → %s", output_path)
    return output_path


def write_manifest(output_dir: Path, data: dict) -> Path:
    """Write a JSON manifest describing all outputs in *output_dir*."""
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info("Manifest → %s", manifest_path)
    return manifest_path
