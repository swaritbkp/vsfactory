"""Benchmark prompts (SPEC 12.1): system instruction, consumption-resolution image, question text with
the answer-format line, and the request-body builder shared by the OpenRouter client and replay keys."""

from __future__ import annotations

import base64
import hashlib
import io
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from knotwork.knotcore.contracts import CONSUMPTION_PX
from knotwork.knotcore.seeds import array_digest

SYSTEM = (
    "You are taking a visual reasoning test. Answer from the image only. "
    "End your reply with a single line 'ANSWER: <answer>'."
)
FORMAT = {
    "label": "ANSWER: <label>",
    "int": "ANSWER: <integer>",
    "float": "ANSWER: <number>",
    "zone": "ANSWER: <green|amber|red>",
    "premise": "ANSWER: <yes|no>; <number>",
}


def with_format(text: str, answer_type: str) -> str:
    """Question text ending with the SPEC 13.3 answer-format line (added only if missing)."""
    line = f"End your reply with a final line of the form '{FORMAT[answer_type]}'."
    return text if line in text else f"{text.rstrip()}\n{line}"


def trace_only(socket: str, k: int) -> str:
    """Composition sub-task 1 (SPEC 12.3): TRACE only, label answer, chance 1/k."""
    return with_format(
        f"Which gauge (G1-G{k}) is the cable from socket {socket} plugged into?", "label"
    )


def read_only(gauge_label: str) -> str:
    """Composition sub-task 2 (SPEC 12.3): READ only, numeric answer."""
    return with_format(f"What value does gauge {gauge_label} read?", "float")


def image_png(beauty: Path) -> bytes:
    """beauty.png resized to the 768-px consumption resolution, PNG-encoded (deterministic)."""
    with Image.open(beauty) as im:
        rgb = im.convert("RGB")
        s = CONSUMPTION_PX / max(rgb.size)
        small = rgb.resize(
            (round(rgb.size[0] * s), round(rgb.size[1] * s)), Image.Resampling.LANCZOS
        )
    buf = io.BytesIO()
    small.save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def image_sha256(png: bytes) -> str:
    return hashlib.sha256(png).hexdigest()


def image_id(path: Path) -> str:
    """Platform-neutral image identity: digest of the decoded RGB pixels of the *source* file (D-35).
    Re-encoded PNG bytes and Lanczos-resized pixels differ between Windows and Linux builds."""
    with Image.open(path) as im:
        return array_digest(np.asarray(im.convert("RGB"), dtype=np.uint8))


def png_pixel_id(png: bytes) -> str:
    """Fallback identity for raw PNG bytes (decoded pixels, not encoder output); undecodable bytes
    fall back to their sha256."""
    try:
        with Image.open(io.BytesIO(png)) as im:
            return array_digest(np.asarray(im.convert("RGB"), dtype=np.uint8))
    except (OSError, ValueError):
        return "sha256:" + image_sha256(png)


def messages(png: bytes, prompt: str, detail: str = "high") -> list[dict[str, Any]]:
    url = "data:image/png;base64," + base64.b64encode(png).decode("ascii")
    return [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": url, "detail": detail}},
                {"type": "text", "text": prompt},
            ],
        },
    ]


def request_body(slot: dict[str, Any], defaults: dict[str, Any], routing: dict[str, Any],
                 png: bytes, prompt: str) -> dict[str, Any]:  # fmt: skip
    """OpenRouter chat-completions body (SPEC 12.4); no secrets inside."""
    body: dict[str, Any] = {
        "model": slot["slug"],
        "messages": messages(png, prompt, str(defaults.get("image_detail", "high"))),
        "temperature": float(defaults.get("temperature", 0.0)),
        "max_tokens": int(slot.get("max_tokens") or defaults.get("max_tokens", 512)),
        "provider": dict(
            routing.get("provider", {"data_collection": "deny", "allow_fallbacks": True})
        ),
        "usage": dict(routing.get("usage", {"include": True})),
    }
    effort = slot.get("reasoning_effort", defaults.get("reasoning_effort"))
    if effort:
        body["reasoning"] = {"effort": str(effort)}
    return body


__all__ = ["FORMAT", "SYSTEM", "image_png", "image_id", "image_sha256", "png_pixel_id", "messages", "read_only", "request_body",
           "trace_only", "with_format"]  # fmt: skip
