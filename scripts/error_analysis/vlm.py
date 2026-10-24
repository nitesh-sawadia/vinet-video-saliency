"""Vision-LLM confirmation of failure hypotheses via the Amagi LLM gateway."""

from __future__ import annotations

import base64
import io
import json
import os
import re
import urllib.error
import urllib.request

from PIL import Image

from hypotheses import HYPOTHESIS_IDS, HYPOTHESIS_BY_ID


def load_env(dotenv_path: str | None = None) -> dict:
    """Load KEY=VALUE pairs from .env without requiring python-dotenv.

    Repo .env values for OPENAI_API_KEY / BASE_URL win over empty process env.
    """
    env = dict(os.environ)
    path = dotenv_path or os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", ".env")
    )
    if not os.path.isfile(path):
        return env
    file_vals = {}
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("["):
                continue
            key, value = line.split("=", 1)
            file_vals[key.strip()] = value.strip().strip('"').strip("'")
    for key, value in file_vals.items():
        if key not in env or key in ("OPENAI_API_KEY", "BASE_URL"):
            env[key] = value
    return env


def _encode_bytes(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _encode_image(path: str) -> str:
    with open(path, "rb") as handle:
        return _encode_bytes(handle.read())


def _mime_for(path: str) -> str:
    lower = path.lower()
    if lower.endswith(".webp"):
        return "image/webp"
    if lower.endswith(".jpg") or lower.endswith(".jpeg"):
        return "image/jpeg"
    if lower.endswith(".gif"):
        return "image/gif"
    return "image/png"


def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def _load_clip_frames(clip_path: str) -> list[Image.Image]:
    """Load all frames from an animated WebP/GIF clip (expects 32 panels)."""
    image = Image.open(clip_path)
    frames = []
    try:
        while True:
            frames.append(image.copy().convert("RGB"))
            image.seek(image.tell() + 1)
    except EOFError:
        pass
    if not frames:
        frames = [image.convert("RGB")]
    return frames


def _jpeg_data_url(pil_image: Image.Image, quality: int = 75) -> str:
    buf = io.BytesIO()
    pil_image.save(buf, format="JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64,%s" % _encode_bytes(buf.getvalue())


def _image_content_parts(
    clip_path: str | None,
    panel_path: str | None,
) -> tuple[list[dict], str]:
    """Build multimodal content parts: prefer all 32 clip frames."""
    parts: list[dict] = []
    source = "none"
    if clip_path and os.path.isfile(clip_path):
        frames = _load_clip_frames(clip_path)
        source = "clip:%s frames=%d" % (clip_path, len(frames))
        # Keep every frame so the VLM can judge temporal consistency.
        for index, frame in enumerate(frames):
            parts.append(
                {
                    "type": "text",
                    "text": "Frame %d/%d of the 32-frame comparison clip "
                    "(left=RGB, middle=GT overlay, right=prediction overlay)."
                    % (index + 1, len(frames)),
                }
            )
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _jpeg_data_url(frame)},
                }
            )
        return parts, source

    if panel_path and os.path.isfile(panel_path):
        source = "panel:%s" % panel_path
        parts.append(
            {
                "type": "text",
                "text": (
                    "Only the last-frame still is available (full 32-frame clip missing). "
                    "Do not confirm temporal_inconsistency unless metrics alone force it."
                ),
            }
        )
        parts.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:%s;base64,%s"
                    % (_mime_for(panel_path), _encode_image(panel_path))
                },
            }
        )
        return parts, source

    raise FileNotFoundError("Need clip_path or panel_path for VLM input")


def confirm_hypotheses(
    sample: dict,
    proposals: list[dict],
    *,
    clip_path: str | None = None,
    panel_path: str | None = None,
    model: str = "vertex_ai/gemini-2.5-flash",
    base_url: str | None = None,
    api_key: str | None = None,
    timeout: float = 180.0,
) -> dict:
    """Ask the VLM which catalog hypotheses are supported by the full clip.

    Prefer ``clip_path`` (animated 32-frame WebP); falls back to ``panel_path``.
    """
    env = load_env()
    base_url = (base_url or env.get("BASE_URL") or "").rstrip("/")
    api_key = api_key or env.get("OPENAI_API_KEY")
    if not base_url or not api_key:
        raise SystemExit(
            "VLM needs BASE_URL and OPENAI_API_KEY in the environment or repo .env"
        )

    media_parts, media_source = _image_content_parts(clip_path, panel_path)

    catalog = [
        {
            "id": hid,
            "title": HYPOTHESIS_BY_ID[hid]["title"],
            "description": HYPOTHESIS_BY_ID[hid]["description"],
        }
        for hid in HYPOTHESIS_IDS
        if hid != "unexplained"
    ]
    proposed_ids = [p["id"] for p in proposals]
    prompt = {
        "task": (
            "You are reviewing a sports video-saliency failure over a full "
            "32-frame clip. Each attached image is one timestep as a 1x3 panel: "
            "left=RGB frame, middle=ground-truth saliency overlay for that frame, "
            "right=model prediction overlay. The model predicts ONE static map for "
            "the whole window (shown on the right of every frame)."
        ),
        "media_source": media_source,
        "sample": {
            "sport": sample.get("sport"),
            "game": sample.get("game"),
            "window": sample.get("window"),
            "cc": sample.get("cc"),
            "kldiv": sample.get("kldiv"),
            "loss": sample.get("loss"),
            "features": {
                "gt_peak_displacement_norm": (sample.get("features") or {}).get(
                    "gt_peak_displacement_norm"
                ),
                "gt_peak_path_norm": (sample.get("features") or {}).get(
                    "gt_peak_path_norm"
                ),
                "gt_temporal_corr_mean": (sample.get("features") or {}).get(
                    "gt_temporal_corr_mean"
                ),
                "clip_motion_mean": (sample.get("features") or {}).get("clip_motion_mean"),
                "peak_distance_norm": (sample.get("features") or {}).get(
                    "peak_distance_norm"
                ),
            },
            "proposed_hypothesis_ids": proposed_ids,
            "proposal_evidence": proposals,
        },
        "catalog": catalog,
        "instructions": [
            "Watch all 32 frames before deciding — especially for temporal_inconsistency.",
            "Confirm temporal_inconsistency when GT attention clearly moves across the "
            "clip while the prediction stays fixed on one place.",
            "Confirm only hypotheses clearly supported by the clip and metrics.",
            "You may confirm catalog ids that were not proposed if visually obvious.",
            "Prefer 1-2 confirmed ids; use none if nothing fits.",
            "Do not invent ids outside the catalog.",
            "Return strict JSON with keys confirmed (list of {id, confidence 0-1, rationale}) "
            "and notes (short string).",
        ],
    }

    content = [{"type": "text", "text": json.dumps(prompt, indent=2)}]
    content.extend(media_parts)

    body = {
        "model": model,
        "temperature": 0.1,
        "messages": [{"role": "user", "content": content}],
    }
    request = urllib.request.Request(
        base_url + "/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": "Bearer %s" % api_key,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError("VLM HTTP %s: %s" % (exc.code, detail[:500])) from exc

    raw = payload["choices"][0]["message"]["content"]
    if isinstance(raw, list):
        raw = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in raw
        )
    parsed = _extract_json(raw)
    confirmed = []
    for item in parsed.get("confirmed") or []:
        hid = item.get("id")
        if hid not in HYPOTHESIS_BY_ID or hid == "unexplained":
            continue
        confirmed.append(
            {
                "id": hid,
                "confidence": float(item.get("confidence", 0.5)),
                "rationale": str(item.get("rationale", "")).strip(),
            }
        )
    confirmed.sort(key=lambda x: x["confidence"], reverse=True)
    confirmed_ids = {c["id"] for c in confirmed}
    rejected = [p for p in proposals if p["id"] not in confirmed_ids]
    return {
        "confirmed": confirmed,
        "rejected_proposals": rejected,
        "notes": str(parsed.get("notes", "")).strip(),
        "raw": raw,
        "model": model,
        "media_source": media_source,
    }
