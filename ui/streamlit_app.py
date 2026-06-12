"""
Streamlit web UI for car2asset.

Run with:
    streamlit run ui/streamlit_app.py
"""

from __future__ import annotations

import io
import sys
import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image

# Allow importing car2asset from src/ without installing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

st.set_page_config(
    page_title="Car → 3D Asset",
    page_icon="🚗",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar — settings
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("⚙️ Settings")
    mode = st.selectbox("Output mode", ["both", "realistic", "cartoon"], index=0)
    quality = st.selectbox("Quality", ["fast", "high", "ultra"], index=1)
    export_fbx = st.checkbox("Also export FBX (requires Blender)", value=False)
    remove_bg = st.checkbox("Remove background", value=True)
    st.markdown("---")
    st.markdown(
        "**Quality guide**\n"
        "- `fast` — TripoSR, ~30 s, 6 GB VRAM\n"
        "- `high` — Hunyuan3D, ~5 min, 10 GB VRAM\n"
        "- `ultra` — Hunyuan3D + PBR texture, ~15 min, 29 GB VRAM"
    )

# ---------------------------------------------------------------------------
# Main — image upload
# ---------------------------------------------------------------------------

st.title("🚗 Car → 3D Asset Generator")
st.caption("Upload 4 images of a car and generate realistic and cartoon 3D assets.")

cols = st.columns(4)
view_labels = ["Front", "Rear", "Left", "Right"]
uploaded: dict[str, bytes | None] = {}

for col, label in zip(cols, view_labels):
    with col:
        f = col.file_uploader(label, type=["jpg", "jpeg", "png", "webp"], key=label.lower())
        uploaded[label.lower()] = f.read() if f else None
        if f:
            col.image(Image.open(io.BytesIO(uploaded[label.lower()])), use_column_width=True)

all_uploaded = all(v is not None for v in uploaded.values())

# ---------------------------------------------------------------------------
# Generate button
# ---------------------------------------------------------------------------

if st.button("Generate 3D Asset", type="primary", disabled=not all_uploaded):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        input_dir = tmpdir / "input"
        output_dir = tmpdir / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        for view, data in uploaded.items():
            (input_dir / f"{view}.jpg").write_bytes(data)

        with st.spinner("Processing … this may take several minutes."):
            try:
                from car2asset.preprocess import prepare_images, save_prepared_images
                from car2asset.reconstruct import generate_realistic_mesh
                from car2asset.stylize_cartoon import generate_cartoon_mesh
                from car2asset.preview import render_preview

                images = prepare_images(input_dir, remove_bg=remove_bg)
                save_prepared_images(images, output_dir)

                realistic_path = output_dir / "car_realistic.glb"
                cartoon_path = output_dir / "car_cartoon.glb"

                if mode in ("realistic", "both"):
                    generate_realistic_mesh(images, realistic_path, quality=quality)
                    render_preview(realistic_path, output_dir / "preview_realistic.png")

                if mode in ("cartoon", "both"):
                    src = realistic_path if realistic_path.exists() else None
                    if src is None:
                        src = output_dir / "_base.glb"
                        generate_realistic_mesh(images, src, quality="fast")
                    generate_cartoon_mesh(src, cartoon_path, reference_images=images)
                    render_preview(cartoon_path, output_dir / "preview_cartoon.png")

                st.success("Generation complete!")

                # ── Preview images ────────────────────────────────────────
                prev_cols = st.columns(2)
                for idx, (name, prev_file, glb_file) in enumerate([
                    ("Realistic", output_dir / "preview_realistic.png", realistic_path),
                    ("Cartoon", output_dir / "preview_cartoon.png", cartoon_path),
                ]):
                    if prev_file.exists():
                        with prev_cols[idx]:
                            st.image(str(prev_file), caption=f"{name} preview", use_column_width=True)

                # ── Download buttons ──────────────────────────────────────
                st.subheader("Downloads")
                dl_cols = st.columns(4)
                dl_idx = 0
                for fname in ["car_realistic.glb", "car_cartoon.glb", "car_realistic.fbx", "car_cartoon.fbx"]:
                    fpath = output_dir / fname
                    if fpath.exists():
                        with dl_cols[dl_idx % 4]:
                            st.download_button(
                                label=f"⬇ {fname}",
                                data=fpath.read_bytes(),
                                file_name=fname,
                                mime="model/gltf-binary" if fname.endswith(".glb") else "application/octet-stream",
                            )
                        dl_idx += 1

            except Exception as exc:
                st.error(f"Pipeline error: {exc}")
                st.exception(exc)

elif not all_uploaded:
    st.info("Upload all four views (front, rear, left, right) to enable generation.")
