"""Image preprocessing: validation, background removal, view labelling."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

EXPECTED_VIEWS = ("front", "rear", "left", "right")
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
TARGET_SIZE = (512, 512)  # standard input size for most 3D backends


class ImageValidationError(ValueError):
    pass


def _find_view_image(input_dir: Path, view: str) -> Path:
    """Locate an image file for *view* inside *input_dir*."""
    for ext in SUPPORTED_EXTS:
        candidate = input_dir / f"{view}{ext}"
        if candidate.exists():
            return candidate
    raise ImageValidationError(
        f"Missing image for view '{view}' in {input_dir}. "
        f"Expected a file named '{view}.jpg', '{view}.png', etc."
    )


def validate_input_dir(input_dir: Path) -> dict[str, Path]:
    """Check that *input_dir* contains all four required view images."""
    if not input_dir.is_dir():
        raise ImageValidationError(f"Input directory does not exist: {input_dir}")

    paths: dict[str, Path] = {}
    missing: list[str] = []
    for view in EXPECTED_VIEWS:
        try:
            paths[view] = _find_view_image(input_dir, view)
        except ImageValidationError:
            missing.append(view)

    if missing:
        raise ImageValidationError(
            f"Missing views: {missing}. "
            f"Place images named front/rear/left/right (jpg or png) in {input_dir}."
        )

    logger.info("Found all four view images in %s", input_dir)
    return paths


def remove_background(image: Image.Image) -> Image.Image:
    """Strip the background and return an RGBA image with alpha mask."""
    try:
        from rembg import remove
    except ImportError as exc:
        raise RuntimeError(
            "rembg is not installed. Run: pip install rembg onnxruntime"
        ) from exc

    return remove(image)


def prepare_images(
    input_dir: Path,
    *,
    remove_bg: bool = True,
    target_size: tuple[int, int] = TARGET_SIZE,
) -> dict[str, Image.Image]:
    """
    Validate, optionally background-strip, and resize all four car view images.

    Returns a dict mapping view name → PIL RGBA image.
    """
    view_paths = validate_input_dir(input_dir)
    prepared: dict[str, Image.Image] = {}

    for view, path in view_paths.items():
        logger.info("Processing %s view: %s", view, path)
        img = Image.open(path).convert("RGBA")

        if remove_bg:
            logger.info("  Removing background from %s view …", view)
            img = remove_background(img)

        img = img.resize(target_size, Image.LANCZOS)
        prepared[view] = img
        logger.info("  Done: %s (%dx%d RGBA)", view, *img.size)

    return prepared


def save_prepared_images(images: dict[str, Image.Image], output_dir: Path) -> dict[str, Path]:
    """Write prepared images to *output_dir*/prepared/ and return their paths."""
    prepared_dir = output_dir / "prepared"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Path] = {}
    for view, img in images.items():
        dest = prepared_dir / f"{view}.png"
        img.save(dest)
        saved[view] = dest
    return saved


def images_to_array_list(images: dict[str, Image.Image]) -> list[np.ndarray]:
    """Convert view dict to an ordered list of RGBA numpy arrays (front/rear/left/right)."""
    return [np.array(images[v]) for v in EXPECTED_VIEWS if v in images]
