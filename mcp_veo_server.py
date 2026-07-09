from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

import os
import requests


load_dotenv()

ACEDATA_VEO_API_URL = "https://api.acedata.cloud/veo/videos"
DEFAULT_OUTPUT_DIR = Path("creative_video/output")

VeoAction = Literal["text2video", "image2video"]

mcp = FastMCP("veo-creative-video-mcp")


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
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _find_video_url(data: Any) -> str | None:
    if isinstance(data, dict):
        for key in ("video_url", "url", "output_url", "download_url"):
            value = data.get(key)
            if isinstance(value, str) and value.startswith("http"):
                return value
        for value in data.values():
            found = _find_video_url(value)
            if found:
                return found

    if isinstance(data, list):
        for item in data:
            found = _find_video_url(item)
            if found:
                return found

    return None


def _download_file(url: str, output_path: Path) -> None:
    response = requests.get(url, timeout=180)
    response.raise_for_status()
    output_path.write_bytes(response.content)


def _generate_video(
    prompt: str,
    action: VeoAction = "text2video",
    output_path: str | None = None,
    image_url: str | None = None,
    timeout_seconds: int = 900,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "action": action,
        "prompt": prompt,
    }

    if image_url:
        payload["image_url"] = image_url

    try:
        response = requests.post(
            ACEDATA_VEO_API_URL,
            json=payload,
            headers=_headers(),
            timeout=timeout_seconds,
        )
        response.raise_for_status()
    except requests.Timeout:
        return {
            "success": False,
            "error": {
                "code": "timeout",
                "message": f"Veo request exceeded {timeout_seconds} seconds.",
            },
            "hint": "Video generation can take several minutes. Try a higher timeout_seconds value.",
        }
    except requests.HTTPError as error:
        return {
            "success": False,
            "error": {
                "code": "http_error",
                "message": str(error),
                "response": response.text,
            },
        }

    data = response.json()

    video_url = _find_video_url(data)
    if video_url:
        target = _safe_output_path(output_path, "veo-video.mp4")
        _download_file(video_url, target)
        data["local_path"] = str(target.resolve())

    return data


@mcp.tool()
def generate_video_from_prompt(
    prompt: str,
    output_path: str | None = "creative_video/output/veo-demo.mp4",
    timeout_seconds: int = 900,
) -> dict[str, Any]:
    """Gera um video com Veo a partir de texto e baixa o MP4 quando a API retornar URL."""
    return _generate_video(
        prompt=prompt,
        action="text2video",
        output_path=output_path,
        timeout_seconds=timeout_seconds,
    )


@mcp.tool()
def generate_project_demo_clip(
    output_path: str = "creative_video/output/maritaca-agentic-rag-demo.mp4",
    timeout_seconds: int = 900,
) -> dict[str, Any]:
    """Gera um clipe chamativo para o Maritaca Hybrid Graph Agentic RAG."""
    prompt = (
        "A premium tech demo video for a Brazilian AI engineering project named "
        "Maritaca Hybrid Graph Agentic RAG. Dark observability dashboard, neon "
        "green execution graph, nodes lighting up in sequence: User Query, "
        "Planner Agent, CAG Memory, RAG Memory, Judge Agent, Final Answer. "
        "Smooth cinematic camera movement, glowing route highlights, high contrast, "
        "professional portfolio style, sharp interface, no distorted text."
    )
    return _generate_video(
        prompt=prompt,
        action="text2video",
        output_path=output_path,
        timeout_seconds=timeout_seconds,
    )


@mcp.tool()
def generate_video_from_image(
    prompt: str,
    image_url: str,
    output_path: str | None = "creative_video/output/veo-image-demo.mp4",
    timeout_seconds: int = 900,
) -> dict[str, Any]:
    """Gera video com Veo usando uma imagem de referencia por URL."""
    return _generate_video(
        prompt=prompt,
        action="image2video",
        image_url=image_url,
        output_path=output_path,
        timeout_seconds=timeout_seconds,
    )


if __name__ == "__main__":
    mcp.run()
