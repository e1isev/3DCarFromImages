# Car → 3D Asset Generator

Convert 4 car photos (front, rear, left, right) into production-ready realistic
and cartoon 3D assets (`.glb` / `.fbx`) using AI-assisted 3D reconstruction.

---

## Quick start

```bash
# 1. Install dependencies
pip install -e ".[api,ui]"

# 2. Place your car images
inputs/example_car/
  front.jpg
  rear.jpg
  left.jpg
  right.jpg

# 3. Run the pipeline
python -m car2asset.run \
  --input  inputs/example_car \
  --mode   both \
  --quality high \
  --output outputs/example_car
```

### No GPU? Use the Colab notebook

Local reconstruction (TripoSR / Hunyuan3D) needs a CUDA GPU with several GB of VRAM.
If your machine doesn't have one (e.g. a laptop with integrated graphics), run the
reconstruction step on a free Google Colab GPU instead:

1. Open [`notebooks/colab_reconstruct.ipynb`](notebooks/colab_reconstruct.ipynb) in Colab.
2. Set runtime to a `T4 GPU`, run all cells, upload your car photo, and download
   `car_realistic.glb`.
3. Feed it back into the local pipeline with `--source-mesh`, which skips
   reconstruction and only runs the lightweight, CPU-friendly steps (cartoon
   stylisation, preview rendering, export) on your machine:

```bash
python -m car2asset.run \
  --input inputs/example_car \
  --mode both \
  --quality fast \
  --source-mesh path/to/car_realistic.glb
```

### Outputs

```
outputs/example_car/
  car_realistic.glb        ← PBR-textured mesh
  car_cartoon.glb          ← Cel-shaded toon mesh
  preview_realistic.png
  preview_cartoon.png
  textures/                ← PBR maps (ultra quality only)
    basecolor.png
    roughness.png
    metallic.png
    normal.png
  manifest.json
```

---

## Modes

| Flag | Description |
|------|-------------|
| `--mode realistic` | High-detail mesh with PBR materials |
| `--mode cartoon` | Stylised mesh with toon shader and outline |
| `--mode both` | Both outputs from the same pipeline run |

## Quality presets

| Preset | Backend | VRAM | Speed |
|--------|---------|------|-------|
| `--quality fast` | TripoSR (single image) | ~6 GB | ~30 s |
| `--quality high` | Hunyuan3D-2mv (multi-view) | ~10 GB | ~5 min |
| `--quality ultra` | Hunyuan3D-2mv + PBR texture | ~29 GB | ~15 min |

---

## Web UI (Streamlit)

```bash
streamlit run ui/streamlit_app.py
```

Upload all four views, pick your quality, download your assets.

## REST API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

```http
POST /generate
  (multipart) front, rear, left, right images
  (query)     mode=both&quality=high&export_fbx=false
→ { "job_id": "...", "status": "queued" }

GET /status/{job_id}
→ { "status": "done", "outputs": { "realistic_glb": "car_realistic.glb", ... } }

GET /download/{job_id}/car_realistic.glb
→ binary GLB file
```

## FBX export (Unreal / Unity)

Install [Blender ≥ 3.6](https://www.blender.org/download/) and set the path:

```bash
# .env
BLENDER_PATH=/usr/bin/blender
```

Then add `--fbx` to the CLI:

```bash
python -m car2asset.run --input inputs/example_car --mode both --quality high --fbx
```

## Docker

```bash
# Build
docker build -t car2asset:latest -f docker/Dockerfile .

# Run API (GPU required)
docker run --gpus all \
  -p 8000:8000 \
  -v $(pwd)/inputs:/app/inputs \
  -v $(pwd)/outputs:/app/outputs \
  car2asset:latest
```

---

## Backends

### Hunyuan3D (recommended — `high` / `ultra`)

Multi-view shape model from Tencent. Uses all 4 car images for better
front/rear/left/right consistency.

```bash
pip install git+https://github.com/Tencent/Hunyuan3D-2.git
```

> **VRAM requirements:** ~10 GB (shape), ~21 GB (texture), ~29 GB (both).

**Pre-download the weights.** `from_pretrained` does not reliably auto-fetch
from the Hub and may fail with `Model path ... not found`. Download the
snapshot explicitly first, then point the pipeline at the local folder via
`.env`:

```bash
pip install -U huggingface_hub
huggingface-cli download tencent/Hunyuan3D-2mv --local-dir models/Hunyuan3D-2mv
huggingface-cli download tencent/Hunyuan3D-2   --local-dir models/Hunyuan3D-2   # only needed for --quality ultra
```

```bash
# .env
HUNYUAN3D_SHAPE_MODEL_PATH=models/Hunyuan3D-2mv
HUNYUAN3D_TEXTURE_MODEL_PATH=models/Hunyuan3D-2

# Hunyuan3D-2mv stores DiT weights under "hunyuan3d-dit-v2-mv", not the
# library's default "hunyuan3d-dit-v2-0" (that default is for the
# single-view Hunyuan3D-2 repo). Set this or the pipeline won't find the
# local snapshot and will try (and fail) to re-download from the Hub.
HUNYUAN3D_SHAPE_SUBFOLDER=hunyuan3d-dit-v2-mv
```

### TripoSR (`fast`)

Single-image fast reconstruction. Good for iteration and preview.

```bash
pip install git+https://github.com/vast-ai-research/TripoSR.git
```

### Stable Fast 3D (`fast`)

Single-image with UV/texture baking, minimal VRAM.

```bash
pip install git+https://github.com/Stability-AI/stable-fast-3d.git
```

---

## Pipeline overview

```
4 car photos
  │
  ├─ validate (front / rear / left / right)
  ├─ background removal  (rembg / onnxruntime)
  ├─ resize → 512×512 RGBA
  │
  ├─ [realistic path]
  │    ├─ Hunyuan3D-2mv shape generation
  │    ├─ Hunyuan3D-Paint PBR textures  (ultra only)
  │    ├─ Blender mesh cleanup          (optional)
  │    └─ export  car_realistic.glb / .fbx
  │
  └─ [cartoon path]
       ├─ simplify geometry (40% face reduction)
       ├─ Laplacian smooth
       ├─ dominant colour extraction
       ├─ toon PBR material (bright, low metallic)
       ├─ inverted-hull outline mesh
       └─ export  car_cartoon.glb / .fbx
```

---

## Project structure

```
car-to-3d-asset/
  src/car2asset/
    __init__.py         package version
    config.py           settings (env-driven)
    run.py              CLI entry point
    preprocess.py       image validation + background removal
    reconstruct.py      3D mesh generation (TripoSR / Hunyuan3D)
    stylize_cartoon.py  cartoon stylisation pass
    export_asset.py     GLB / FBX export helpers
    preview.py          preview PNG renderer
  blender/
    cleanup_mesh.py     Blender headless mesh cleanup
    cartoon_shader.py   Blender cel-shade material
    export_fbx.py       Blender GLB → FBX converter
  api/
    main.py             FastAPI REST server
  ui/
    streamlit_app.py    Streamlit web UI
  docker/
    Dockerfile          CUDA-enabled container
  inputs/example_car/  sample input folder
  outputs/             generated assets (git-ignored)
```

## Git LFS

Large binary files (`.glb`, `.fbx`, `.blend`, model weights) are tracked
via Git LFS. Run `git lfs install` before cloning if you want to pull
example output assets.

---

## Notes

- 4 photos are enough for AI-generated assets but not for scan-grade photogrammetry. For near-perfect geometry, use 20–60 images around the car.
- For best results, shoot outdoors with even lighting, no strong shadows, full car visible in each frame.
- The cartoon pass reuses the realistic mesh — it does not require a second reconstruction.

## Roadmap

- [ ] Wheel/door bone separation for animation
- [ ] Suspension bounce idle animation
- [ ] Drive-forward animation clip
- [ ] Unreal Engine 5 import guide
- [ ] ComfyUI node integration

## License

MIT
