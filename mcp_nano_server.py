from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

import os
import requests


PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=PROJECT_DIR / ".env")

ACEDATA_API_URL = "https://api.acedata.cloud/nano-banana/images"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "creative_video" / "assets"

NanoBananaModel = Literal["nano-banana", "nano-banana-2", "nano-banana-pro"]
AspectRatio = Literal["1:1", "4:3", "3:4", "16:9", "9:16"]

mcp = FastMCP("nano-banana-creative-mcp")


def _api_token() -> str:
    token = os.getenv("ACEDATACLOUD_API_TOKEN")
    if not token:
        raise RuntimeError("ACEDATACLOUD_API_TOKEN nao encontrado no .env")
    return token


def _headers() -> dict[str, str]:
    return {
        "authorization": f"Bearer {_api_token()}",
        "content-type": "application/json",
    }


def _safe_output_path(output_path: str | None, fallback_name: str) -> Path:
    path = Path(output_path) if output_path else DEFAULT_OUTPUT_DIR / fallback_name
    if not path.is_absolute():
        path = PROJECT_DIR / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _download_file(url: str, output_path: Path) -> None:
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    output_path.write_bytes(response.content)


def _generate_image(
    prompt: str,
    model: NanoBananaModel = "nano-banana",
    aspect_ratio: AspectRatio = "16:9",
    output_path: str | None = None,
) -> dict[str, Any]:
    payload = {
        "action": "generate",
        "model": model,
        "prompt": prompt,
        "aspect_ratio": aspect_ratio,
    }

    response = requests.post(
        ACEDATA_API_URL,
        json=payload,
        headers=_headers(),
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()

    if not data.get("success"):
        return data

    images = data.get("data") or []
    if images:
        image_url = images[0].get("image_url")
        if image_url:
            target = _safe_output_path(output_path, "nano-banana-image.png")
            _download_file(image_url, target)
            data["local_path"] = str(target.resolve())

    return data


@mcp.tool()
def generate_image_asset(
    prompt: str,
    output_path: str | None = None,
    model: NanoBananaModel = "nano-banana",
    aspect_ratio: AspectRatio = "16:9",
) -> dict[str, Any]:
    """Gera uma imagem com Nano Banana e salva localmente quando houver image_url."""
    return _generate_image(
        prompt=prompt,
        model=model,
        aspect_ratio=aspect_ratio,
        output_path=output_path,
    )


@mcp.tool()
def generate_project_cover(
    output_path: str = "creative_video/assets/maritaca-cover.png",
    model: NanoBananaModel = "nano-banana",
) -> dict[str, Any]:
    """Gera uma capa 16:9 para o projeto Maritaca Hybrid Graph Agentic RAG."""
    prompt = (
        "Cinematic 16:9 hero image for a Brazilian AI engineering project named "
        '"Maritaca Hybrid Graph Agentic RAG". Dark observability dashboard, neon '
        "green agentic execution graph, glowing nodes labeled Planner Agent, CAG "
        "Memory, RAG Memory, Judge Agent, Final Answer. Premium tech portfolio "
        "aesthetic, high contrast, sharp UI, subtle Brazilian green accents."
    )
    return _generate_image(
        prompt=prompt,
        model=model,
        aspect_ratio="16:9",
        output_path=output_path,
    )


@mcp.tool()
def create_video_prompt(scene: str, duration_seconds: int = 8) -> dict[str, str | int]:
    """Cria um prompt cinematografico para animar uma imagem ou cena com Veo."""
    prompt = (
        f"{scene}. Smooth cinematic camera movement, premium AI engineering demo, "
        "dark dashboard lighting, neon green highlights, glowing route animation, "
        "sharp futuristic interface, professional portfolio video, no distorted text."
    )
    return {
        "duration_seconds": duration_seconds,
        "prompt": prompt,
    }


if __name__ == "__main__":
    mcp.run()
