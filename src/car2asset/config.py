"""Runtime configuration loaded from environment / .env file."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


Quality = Literal["fast", "high", "ultra"]
Mode = Literal["realistic", "cartoon", "both"]
Backend = Literal["triposr", "stable-fast-3d", "hunyuan3d"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CAR2ASSET_", env_file=".env", extra="ignore")

    quality: Quality = "high"
    backend: Backend = "hunyuan3d"
    mode: Mode = "both"
    device: str = "cuda"

    # Blender binary used for FBX export and cartoon shader
    blender_path: Path = Field(default=Path("/usr/bin/blender"), alias="BLENDER_PATH")

    # Hugging Face token for gated model downloads
    hf_token: str = Field(default="", alias="HF_TOKEN")

    # Where downloaded model weights are cached
    model_cache_dir: Path = Path("models")

    # Hunyuan3D model location: a HF repo id (e.g. "tencent/Hunyuan3D-2mv") or a
    # local directory if the weights were pre-downloaded with `huggingface-cli download`.
    # Pre-downloading is recommended: Hunyuan3D's from_pretrained does not reliably
    # auto-fetch from the Hub and may raise "Model path ... not found" otherwise.
    hunyuan3d_shape_model_path: str = Field(
        default="tencent/Hunyuan3D-2mv", alias="HUNYUAN3D_SHAPE_MODEL_PATH"
    )
    hunyuan3d_texture_model_path: str = Field(
        default="tencent/Hunyuan3D-2", alias="HUNYUAN3D_TEXTURE_MODEL_PATH"
    )

    @property
    def backend_for_quality(self) -> Backend:
        """Return the best backend for the chosen quality preset."""
        mapping: dict[Quality, Backend] = {
            "fast": "triposr",
            "high": "hunyuan3d",
            "ultra": "hunyuan3d",
        }
        return mapping[self.quality]


settings = Settings()
