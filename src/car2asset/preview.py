"""
Render a 512×512 preview PNG of a GLB mesh.

Primary renderer: pyrender (offscreen, cross-platform)
Fallback:         trimesh built-in scene.save_image() (no display needed)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

PREVIEW_SIZE = (512, 512)
CAMERA_DISTANCE = 2.5
LIGHT_INTENSITY = 3.0


def render_preview(mesh_path: Path, output_path: Path) -> Path:
    """
    Render a front-diagonal view of *mesh_path* and save to *output_path* (PNG).

    Returns *output_path*.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        return _render_pyrender(mesh_path, output_path)
    except Exception as exc:
        logger.warning("pyrender failed (%s); falling back to trimesh renderer.", exc)
        return _render_trimesh_fallback(mesh_path, output_path)


def _render_pyrender(mesh_path: Path, output_path: Path) -> Path:
    import pyrender  # type: ignore[import]
    import trimesh  # type: ignore[import]

    scene_tm = trimesh.load(str(mesh_path), force="scene")

    # Build pyrender scene
    scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3], bg_color=[0.15, 0.15, 0.15, 1.0])

    for geom in scene_tm.geometry.values():
        mesh_pr = pyrender.Mesh.from_trimesh(geom, smooth=True)
        scene.add(mesh_pr)

    # Position camera
    bounds = scene_tm.bounds
    centre = (bounds[0] + bounds[1]) / 2
    extent = np.linalg.norm(bounds[1] - bounds[0])
    dist = extent * CAMERA_DISTANCE

    camera = pyrender.PerspectiveCamera(yfov=np.pi / 4)
    cam_pose = _look_at(
        eye=centre + np.array([dist * 0.6, dist * 0.4, dist * 0.6]),
        target=centre,
        up=np.array([0, 1, 0]),
    )
    scene.add(camera, pose=cam_pose)

    # Key light
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=LIGHT_INTENSITY)
    scene.add(light, pose=cam_pose)

    r = pyrender.OffscreenRenderer(*PREVIEW_SIZE)
    colour, _ = r.render(scene)
    r.delete()

    from PIL import Image  # type: ignore[import]

    Image.fromarray(colour).save(str(output_path))
    logger.info("Preview (pyrender) → %s", output_path)
    return output_path


def _render_trimesh_fallback(mesh_path: Path, output_path: Path) -> Path:
    import trimesh  # type: ignore[import]

    scene = trimesh.load(str(mesh_path), force="scene")
    png_bytes = scene.save_image(resolution=PREVIEW_SIZE, visible=False)
    with open(output_path, "wb") as f:
        f.write(png_bytes)
    logger.info("Preview (trimesh) → %s", output_path)
    return output_path


def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Build a 4×4 camera pose matrix (column-major OpenGL convention)."""
    z = eye - target
    z /= np.linalg.norm(z)
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    mat = np.eye(4)
    mat[:3, 0] = x
    mat[:3, 1] = y
    mat[:3, 2] = z
    mat[:3, 3] = eye
    return mat
