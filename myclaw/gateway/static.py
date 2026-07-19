from __future__ import annotations

import asyncio
import mimetypes
from pathlib import Path, PurePosixPath
from urllib.parse import unquote

from myclaw.gateway.http import send_response


_WEB_DIST_DIR = Path(__file__).resolve().parents[1] / "web" / "dist"


async def send_web_asset(writer: asyncio.StreamWriter, request_path: str) -> bool:
    decoded_path = unquote(request_path)
    relative_path = "index.html" if decoded_path == "/" else decoded_path.removeprefix("/")
    relative = PurePosixPath(relative_path)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        return False

    candidate = _WEB_DIST_DIR.joinpath(*relative.parts).resolve()
    if not candidate.is_relative_to(_WEB_DIST_DIR) or not candidate.is_file():
        return False

    content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
    if content_type.startswith("text/") or content_type in {"application/javascript", "application/json"}:
        content_type += "; charset=utf-8"
    await send_response(writer, 200, candidate.read_bytes(), content_type)
    return True
