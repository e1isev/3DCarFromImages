"""
FastAPI REST server for car2asset.

POST /generate   — upload 4 images, trigger pipeline, return job ID
GET  /status/{job_id}  — poll job status
GET  /download/{job_id}/{filename} — download output file
GET  /health     — liveness check

Run with:
    uvicorn api.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import uuid
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)
app = FastAPI(title="car2asset API", version="0.1.0")

JOBS_DIR = Path("outputs/jobs")
JOBS_DIR.mkdir(parents=True, exist_ok=True)

JobStatus = Literal["queued", "running", "done", "error"]
_jobs: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class GenerateRequest(BaseModel):
    mode: Literal["realistic", "cartoon", "both"] = "both"
    quality: Literal["fast", "high", "ultra"] = "high"
    export_fbx: bool = False


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    outputs: dict[str, str] = {}
    error: str | None = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/generate", response_model=JobResponse, status_code=202)
async def generate(
    front: UploadFile = File(...),
    rear: UploadFile = File(...),
    left: UploadFile = File(...),
    right: UploadFile = File(...),
    mode: Literal["realistic", "cartoon", "both"] = "both",
    quality: Literal["fast", "high", "ultra"] = "high",
    export_fbx: bool = False,
):
    job_id = str(uuid.uuid4())
    job_dir = JOBS_DIR / job_id
    input_dir = job_dir / "input"
    output_dir = job_dir / "output"
    input_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)

    # Save uploaded files
    for view_name, upload in [("front", front), ("rear", rear), ("left", left), ("right", right)]:
        suffix = Path(upload.filename or "img.jpg").suffix or ".jpg"
        dest = input_dir / f"{view_name}{suffix}"
        with open(dest, "wb") as f:
            shutil.copyfileobj(upload.file, f)

    _jobs[job_id] = {"status": "queued", "outputs": {}, "error": None}
    asyncio.create_task(_run_pipeline(job_id, input_dir, output_dir, mode, quality, export_fbx))
    return JobResponse(job_id=job_id, status="queued")


@app.get("/status/{job_id}", response_model=JobResponse)
async def status(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return JobResponse(job_id=job_id, **job)


@app.get("/download/{job_id}/{filename}")
async def download(job_id: str, filename: str):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    if job["status"] != "done":
        raise HTTPException(status_code=409, detail=f"Job is {job['status']}, not done yet")

    file_path = JOBS_DIR / job_id / "output" / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File {filename} not found")
    return FileResponse(str(file_path), filename=filename)


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------


async def _run_pipeline(
    job_id: str,
    input_dir: Path,
    output_dir: Path,
    mode: str,
    quality: str,
    export_fbx: bool,
) -> None:
    _jobs[job_id]["status"] = "running"
    try:
        # Run in thread pool to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        outputs = await loop.run_in_executor(
            None, _pipeline_sync, input_dir, output_dir, mode, quality, export_fbx
        )
        _jobs[job_id]["status"] = "done"
        _jobs[job_id]["outputs"] = outputs
    except Exception as exc:
        logger.exception("Job %s failed", job_id)
        _jobs[job_id]["status"] = "error"
        _jobs[job_id]["error"] = str(exc)


def _pipeline_sync(
    input_dir: Path,
    output_dir: Path,
    mode: str,
    quality: str,
    export_fbx: bool,
) -> dict[str, str]:
    from car2asset.export_asset import export_fbx as do_export_fbx
    from car2asset.preprocess import prepare_images, save_prepared_images
    from car2asset.preview import render_preview
    from car2asset.reconstruct import generate_realistic_mesh
    from car2asset.stylize_cartoon import generate_cartoon_mesh

    images = prepare_images(input_dir)
    save_prepared_images(images, output_dir)

    outputs: dict[str, str] = {}

    realistic_path = output_dir / "car_realistic.glb"
    if mode in ("realistic", "both"):
        generate_realistic_mesh(images, realistic_path, quality=quality)
        outputs["realistic_glb"] = realistic_path.name
        render_preview(realistic_path, output_dir / "preview_realistic.png")
        outputs["preview_realistic"] = "preview_realistic.png"
        if export_fbx:
            try:
                fbx = output_dir / "car_realistic.fbx"
                do_export_fbx(realistic_path, fbx)
                outputs["realistic_fbx"] = fbx.name
            except Exception as exc:
                logger.warning("FBX export failed: %s", exc)

    if mode in ("cartoon", "both"):
        source = realistic_path if realistic_path.exists() else None
        if source is None:
            source = output_dir / "_base.glb"
            generate_realistic_mesh(images, source, quality="fast")

        cartoon_path = output_dir / "car_cartoon.glb"
        generate_cartoon_mesh(source, cartoon_path, reference_images=images)
        outputs["cartoon_glb"] = cartoon_path.name
        render_preview(cartoon_path, output_dir / "preview_cartoon.png")
        outputs["preview_cartoon"] = "preview_cartoon.png"
        if export_fbx:
            try:
                fbx = output_dir / "car_cartoon.fbx"
                do_export_fbx(cartoon_path, fbx)
                outputs["cartoon_fbx"] = fbx.name
            except Exception as exc:
                logger.warning("FBX export failed: %s", exc)

    return outputs
