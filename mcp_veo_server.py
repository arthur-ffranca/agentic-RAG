from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

import os
import requests
import time


PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=PROJECT_DIR / ".env")

ACEDATA_VEO_API_URL = "https://api.acedata.cloud/veo/videos"
ACEDATA_VEO_TASKS_URL = "https://api.acedata.cloud/veo/tasks"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "creative_video" / "output"

VeoAction = Literal["text2video", "image2video"]
VeoModel = Literal["veo2", "veo2-fast", "veo3", "veo3-fast", "veo31", "veo31-fast"]
AspectRatio = Literal["16:9", "9:16", "4:3", "3:4", "1:1"]

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
    if not path.is_absolute():
        path = PROJECT_DIR / path
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


def _task_is_complete(data: dict[str, Any]) -> bool:
    if _find_video_url(data):
        return True

    response = data.get("response")
    if isinstance(response, dict) and _find_video_url(response):
        return True

    text = str(data).lower()
    return any(state in text for state in ("succeeded", "completed", "failed", "error"))


def _download_file(url: str, output_path: Path) -> None:
    response = requests.get(url, timeout=180)
    response.raise_for_status()
    output_path.write_bytes(response.content)


def _get_task(task_id: str, output_path: str | None = None) -> dict[str, Any]:
    response = requests.post(
        ACEDATA_VEO_TASKS_URL,
        json={"action": "retrieve", "id": task_id},
        headers=_headers(),
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()

    video_url = _find_video_url(data)
    if video_url:
        target = _safe_output_path(output_path, f"{task_id}.mp4")
        _download_file(video_url, target)
        data["local_path"] = str(target.resolve())

    return data


def _generate_video(
    prompt: str,
    action: VeoAction = "text2video",
    output_path: str | None = None,
    image_urls: list[str] | None = None,
    model: VeoModel = "veo2-fast",
    aspect_ratio: AspectRatio = "16:9",
    timeout_seconds: int = 900,
    poll_interval_seconds: int = 20,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "action": action,
        "model": model,
        "prompt": prompt,
        "aspect_ratio": aspect_ratio,
        "async": True,
    }

    if image_urls:
        payload["image_urls"] = image_urls

    try:
        response = requests.post(
            ACEDATA_VEO_API_URL,
            json=payload,
            headers=_headers(),
            timeout=min(timeout_seconds, 180),
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
    task_id = data.get("task_id")

    if task_id and not _find_video_url(data):
        deadline = time.time() + timeout_seconds
        last_task_result: dict[str, Any] = data

        while time.time() < deadline:
            task_response = requests.post(
                ACEDATA_VEO_TASKS_URL,
                json={"action": "retrieve", "id": task_id},
                headers=_headers(),
                timeout=60,
            )
            task_response.raise_for_status()
            last_task_result = task_response.json()

            if _task_is_complete(last_task_result):
                data = last_task_result
                break

            time.sleep(poll_interval_seconds)
        else:
            return {
                "success": False,
                "task_id": task_id,
                "error": {
                    "code": "timeout",
                    "message": f"Veo task did not complete within {timeout_seconds} seconds.",
                },
                "last_response": last_task_result,
            }

    video_url = _find_video_url(data)
    if video_url:
        target = _safe_output_path(output_path, "veo-video.mp4")
        _download_file(video_url, target)
        data["local_path"] = str(target.resolve())

    return data


def _submit_video_task(
    prompt: str,
    action: VeoAction = "text2video",
    image_urls: list[str] | None = None,
    model: VeoModel = "veo2-fast",
    aspect_ratio: AspectRatio = "16:9",
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "action": action,
        "model": model,
        "prompt": prompt,
        "aspect_ratio": aspect_ratio,
        "async": True,
    }

    if image_urls:
        payload["image_urls"] = image_urls

    response = requests.post(
        ACEDATA_VEO_API_URL,
        json=payload,
        headers=_headers(),
        timeout=180,
    )
    response.raise_for_status()
    return response.json()


@mcp.tool()
def generate_video_from_prompt(
    prompt: str,
    output_path: str | None = "creative_video/output/veo-demo.mp4",
    model: VeoModel = "veo2-fast",
    aspect_ratio: AspectRatio = "16:9",
    timeout_seconds: int = 900,
) -> dict[str, Any]:
    """Gera um video com Veo a partir de texto e baixa o MP4 quando a API retornar URL."""
    return _generate_video(
        prompt=prompt,
        action="text2video",
        output_path=output_path,
        model=model,
        aspect_ratio=aspect_ratio,
        timeout_seconds=timeout_seconds,
    )


@mcp.tool()
def submit_video_task(
    prompt: str,
    model: VeoModel = "veo2-fast",
    aspect_ratio: AspectRatio = "16:9",
) -> dict[str, Any]:
    """Submete uma task assíncrona do Veo e retorna task_id sem aguardar o vídeo."""
    return _submit_video_task(
        prompt=prompt,
        action="text2video",
        model=model,
        aspect_ratio=aspect_ratio,
    )


@mcp.tool()
def get_video_task(
    task_id: str,
    output_path: str | None = None,
) -> dict[str, Any]:
    """Consulta uma task do Veo e baixa o MP4 quando o video estiver pronto."""
    return _get_task(task_id=task_id, output_path=output_path)


@mcp.tool()
def generate_project_demo_clip(
    output_path: str = "creative_video/output/maritaca-agentic-rag-demo.mp4",
    model: VeoModel = "veo2-fast",
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
        model=model,
        aspect_ratio="16:9",
        timeout_seconds=timeout_seconds,
    )


@mcp.tool()
def generate_video_from_image(
    prompt: str,
    image_url: str,
    output_path: str | None = "creative_video/output/veo-image-demo.mp4",
    model: VeoModel = "veo2-fast",
    aspect_ratio: AspectRatio = "16:9",
    timeout_seconds: int = 900,
) -> dict[str, Any]:
    """Gera video com Veo usando uma imagem de referencia por URL."""
    return _generate_video(
        prompt=prompt,
        action="image2video",
        image_urls=[image_url],
        output_path=output_path,
        model=model,
        aspect_ratio=aspect_ratio,
        timeout_seconds=timeout_seconds,
    )


if __name__ == "__main__":
    mcp.run()
