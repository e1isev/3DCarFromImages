"""
CLI entry point.

Usage:
    python -m car2asset.run --input inputs/example_car --mode both --quality high
    car2asset --input inputs/example_car --mode realistic --quality fast
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from car2asset.config import Mode, Quality
from car2asset.export_asset import export_fbx, export_glb, write_manifest
from car2asset.preprocess import prepare_images, save_prepared_images
from car2asset.preview import render_preview
from car2asset.reconstruct import generate_realistic_mesh
from car2asset.stylize_cartoon import generate_cartoon_mesh

console = Console()
app = typer.Typer(
    name="car2asset",
    help="Convert 4 car images into realistic and cartoon 3D assets (GLB/FBX).",
    add_completion=False,
)


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        level=level,
    )


@app.command()
def main(
    input_dir: Annotated[Path, typer.Option("--input", "-i", help="Directory with front/rear/left/right images")],
    output_dir: Annotated[Optional[Path], typer.Option("--output", "-o", help="Output directory")] = None,
    mode: Annotated[Mode, typer.Option("--mode", "-m", help="Output mode")] = "both",
    quality: Annotated[Quality, typer.Option("--quality", "-q", help="Quality preset")] = "high",
    export_fbx_flag: Annotated[bool, typer.Option("--fbx/--no-fbx", help="Also export FBX for Unreal/Unity")] = False,
    no_bg_removal: Annotated[bool, typer.Option("--no-bg-removal", help="Skip background removal")] = False,
    source_mesh: Annotated[
        Optional[Path],
        typer.Option(
            "--source-mesh",
            help=(
                "Path to an already-generated realistic .glb (e.g. produced by a cloud GPU "
                "notebook). Skips local reconstruction and uses this mesh directly."
            ),
        ),
    ] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    _setup_logging(verbose)

    if output_dir is None:
        output_dir = Path("outputs") / input_dir.name

    console.print(Panel.fit(
        f"[bold cyan]car2asset[/bold cyan]  mode=[green]{mode}[/green]  "
        f"quality=[yellow]{quality}[/yellow]\n"
        f"input : {input_dir}\noutput: {output_dir}",
        title="3D Car Asset Generator",
    ))

    t0 = time.perf_counter()
    manifest: dict = {"input": str(input_dir), "mode": mode, "quality": quality, "outputs": {}}

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as progress:

        # ── Step 1: Preprocess ──────────────────────────────────────────────
        task = progress.add_task("Preparing images …", total=None)
        images = prepare_images(input_dir, remove_bg=not no_bg_removal)
        saved_images = save_prepared_images(images, output_dir)
        progress.update(task, description="[green]Images prepared[/green]")

        # ── Step 2: Realistic mesh ──────────────────────────────────────────
        if mode in ("realistic", "both"):
            realistic_path = output_dir / "car_realistic.glb"
            if source_mesh is not None:
                task = progress.add_task("Using provided source mesh …", total=None)
                export_glb(source_mesh, realistic_path)
                progress.update(task, description="[green]Source mesh imported[/green]")
            else:
                task = progress.add_task("Generating realistic mesh …", total=None)
                generate_realistic_mesh(images, realistic_path, quality=quality)
            manifest["outputs"]["realistic_glb"] = str(realistic_path)

            if export_fbx_flag:
                task2 = progress.add_task("Exporting realistic FBX …", total=None)
                try:
                    fbx_path = output_dir / "car_realistic.fbx"
                    export_fbx(realistic_path, fbx_path)
                    manifest["outputs"]["realistic_fbx"] = str(fbx_path)
                except Exception as exc:
                    console.print(f"[yellow]FBX export skipped:[/yellow] {exc}")
                progress.update(task2, description="[green]Realistic FBX done[/green]")

            task_prev = progress.add_task("Rendering realistic preview …", total=None)
            prev_path = output_dir / "preview_realistic.png"
            render_preview(realistic_path, prev_path)
            manifest["outputs"]["preview_realistic"] = str(prev_path)
            progress.update(task, description="[green]Realistic mesh done[/green]")
            progress.update(task_prev, description="[green]Preview done[/green]")

        # ── Step 3: Cartoon mesh ────────────────────────────────────────────
        if mode in ("cartoon", "both"):
            source = realistic_path if mode == "both" else None
            if source is None or not source.exists():
                source = output_dir / "_base_mesh.glb"
                if source_mesh is not None:
                    progress.add_task("Using provided source mesh …", total=None)
                    export_glb(source_mesh, source)
                else:
                    # Generate a realistic mesh silently as base even if mode=cartoon
                    progress.add_task("Generating base mesh for cartoon …", total=None)
                    generate_realistic_mesh(images, source, quality="fast")

            task = progress.add_task("Stylising cartoon mesh …", total=None)
            cartoon_path = output_dir / "car_cartoon.glb"
            generate_cartoon_mesh(source, cartoon_path, reference_images=images)
            manifest["outputs"]["cartoon_glb"] = str(cartoon_path)

            if export_fbx_flag:
                task2 = progress.add_task("Exporting cartoon FBX …", total=None)
                try:
                    fbx_path = output_dir / "car_cartoon.fbx"
                    export_fbx(cartoon_path, fbx_path)
                    manifest["outputs"]["cartoon_fbx"] = str(fbx_path)
                except Exception as exc:
                    console.print(f"[yellow]FBX export skipped:[/yellow] {exc}")
                progress.update(task2, description="[green]Cartoon FBX done[/green]")

            task_prev = progress.add_task("Rendering cartoon preview …", total=None)
            prev_path = output_dir / "preview_cartoon.png"
            render_preview(cartoon_path, prev_path)
            manifest["outputs"]["preview_cartoon"] = str(prev_path)
            progress.update(task, description="[green]Cartoon mesh done[/green]")
            progress.update(task_prev, description="[green]Preview done[/green]")

    # ── Manifest ───────────────────────────────────────────────────────────
    manifest["elapsed_seconds"] = round(time.perf_counter() - t0, 1)
    write_manifest(output_dir, manifest)

    console.print(Panel.fit(
        "\n".join(f"  [cyan]{k}[/cyan]: {v}" for k, v in manifest["outputs"].items())
        + f"\n\n[dim]Completed in {manifest['elapsed_seconds']}s[/dim]",
        title="[bold green]Done[/bold green]",
    ))


if __name__ == "__main__":
    app()
