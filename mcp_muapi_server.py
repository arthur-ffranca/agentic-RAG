from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from shutil import which


PROJECT_DIR = Path(__file__).resolve().parent
ENV_PATH = PROJECT_DIR / ".env"


def load_env_file(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        if key and key not in os.environ:
            os.environ[key] = value


def main() -> int:
    load_env_file(ENV_PATH)

    if not os.getenv("MUAPI_API_KEY"):
        for fallback_key in ("MUAPIAPP_API_KEY", "MUAI_API_KEY", "SEEDANCE_API_TOKEN"):
            if os.getenv(fallback_key):
                os.environ["MUAPI_API_KEY"] = os.environ[fallback_key]
                break

    if not os.getenv("MUAPI_API_KEY"):
        print(
            "MUAPI_API_KEY, MUAPIAPP_API_KEY, MUAI_API_KEY, or SEEDANCE_API_TOKEN not found in .env",
            file=sys.stderr,
        )
        return 1

    muapi_command = which("muapi.cmd") or which("muapi") or "muapi"
    return subprocess.call([muapi_command, "mcp", "serve"], cwd=str(PROJECT_DIR))


if __name__ == "__main__":
    raise SystemExit(main())
