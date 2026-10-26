"""Vision-LLM calls for error analysis.

Transport: Vertex AI ``generateContent`` authenticated with the service account
in ``VERTEX_AI_SERVICE_ACCOUNT`` (RS256 JWT -> Google OAuth bearer, cached).
The Amagi LLM gateway (``GEMINI_API_KEY`` + ``BASE_URL``) is a fallback used
only when the service account is absent.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

from PIL import Image

from hypotheses import HYPOTHESIS_IDS, HYPOTHESIS_BY_ID

DEFAULT_VLM_MODEL = "gemini-3.6-flash"
_VERTEX_ENDPOINT = (
    "https://aiplatform.googleapis.com/v1/projects/{project_id}/locations/global/"
    "publishers/google/models/{model}:generateContent"
)
_OAUTH_SCOPE = "https://www.googleapis.com/auth/cloud-platform"
_DEFAULT_TOKEN_URI = "https://oauth2.googleapis.com/token"
_TOKEN_CACHE: dict[str, tuple[str, float]] = {}
_TOKEN_LOCK = threading.Lock()


def load_env(dotenv_path: str | None = None) -> dict:
    """Load KEY=VALUE pairs from .env without requiring python-dotenv.

    Repo .env values for gateway keys / BASE_URL win over empty process env.
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
    prefer_file = ("OPENAI_API_KEY", "GEMINI_API_KEY", "BASE_URL")
    for key, value in file_vals.items():
        if key not in env or key in prefer_file:
            env[key] = value
    return env


def resolve_gateway_key(model: str, env: dict | None = None, api_key: str | None = None) -> str:
    """Pick the gateway key that can call ``model`` (Gemini vs OpenAI/Sol)."""
    if api_key:
        return api_key
    env = env or load_env()
    low = (model or "").lower()
    if any(tok in low for tok in ("gemini", "vertex", "google")):
        key = env.get("GEMINI_API_KEY") or env.get("OPENAI_API_KEY")
    else:
        key = env.get("OPENAI_API_KEY") or env.get("GEMINI_API_KEY")
    if not key:
        raise SystemExit(
            "VLM needs GEMINI_API_KEY or OPENAI_API_KEY in the environment or repo .env"
        )
    return key


def load_service_account(env: dict | None = None) -> dict | None:
    """Parse ``VERTEX_AI_SERVICE_ACCOUNT`` (inline JSON or a path to a JSON file)."""
    env = env if env is not None else load_env()
    raw = (env.get("VERTEX_AI_SERVICE_ACCOUNT") or "").strip()
    if not raw:
        return None
    if not raw.startswith("{") and os.path.isfile(raw):
        with open(raw) as handle:
            raw = handle.read()
    try:
        info = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit("VERTEX_AI_SERVICE_ACCOUNT is not valid JSON") from exc
    missing = [k for k in ("client_email", "private_key", "project_id") if not info.get(k)]
    if missing:
        raise SystemExit("VERTEX_AI_SERVICE_ACCOUNT is missing %s" % missing)
    return info


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def build_service_account_jwt(info: dict, now: float | None = None) -> str:
    """Sign an RS256 JWT assertion for the Google OAuth2 jwt-bearer grant."""
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding

    issued = int(now if now is not None else time.time())
    header = {"alg": "RS256", "typ": "JWT"}
    if info.get("private_key_id"):
        header["kid"] = info["private_key_id"]
    claims = {
        "iss": info["client_email"],
        "scope": _OAUTH_SCOPE,
        "aud": info.get("token_uri") or _DEFAULT_TOKEN_URI,
        "iat": issued,
        "exp": issued + 3600,
    }
    signing_input = "%s.%s" % (
        _b64url(json.dumps(header, separators=(",", ":")).encode("utf-8")),
        _b64url(json.dumps(claims, separators=(",", ":")).encode("utf-8")),
    )
    key = serialization.load_pem_private_key(
        info["private_key"].encode("utf-8"), password=None
    )
    signature = key.sign(signing_input.encode("ascii"), padding.PKCS1v15(), hashes.SHA256())
    return "%s.%s" % (signing_input, _b64url(signature))


def vertex_access_token(info: dict, *, timeout: float = 30.0) -> str:
    """Exchange the service-account JWT for a bearer token; cached until near expiry."""
    cache_key = info["client_email"]
    with _TOKEN_LOCK:
        cached = _TOKEN_CACHE.get(cache_key)
        if cached and cached[1] - 60.0 > time.time():
            return cached[0]
        data = urllib.parse.urlencode(
            {
                "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
                "assertion": build_service_account_jwt(info),
            }
        ).encode("ascii")
        request = urllib.request.Request(
            info.get("token_uri") or _DEFAULT_TOKEN_URI,
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError("Google OAuth HTTP %s: %s" % (exc.code, detail[:300])) from exc
        token = payload.get("access_token")
        if not token:
            raise RuntimeError("Google OAuth response has no access_token")
        expires_at = time.time() + float(payload.get("expires_in") or 3600)
        _TOKEN_CACHE[cache_key] = (token, expires_at)
        return token


def vertex_model_name(model: str) -> str:
    """``vertex_ai/gemini-3.6-flash`` -> ``gemini-3.6-flash`` for the Vertex path."""
    model = (model or DEFAULT_VLM_MODEL).strip()
    return model.split("/", 1)[1] if model.startswith("vertex_ai/") else model


def _gateway_model_name(model: str) -> str:
    model = (model or DEFAULT_VLM_MODEL).strip()
    return model if "/" in model else "vertex_ai/" + model


def _vertex_parts(content: list[dict]) -> list[dict]:
    """OpenAI-style content parts -> Vertex ``parts`` (data URLs become inlineData)."""
    parts = []
    for part in content:
        kind = part.get("type")
        if kind == "text":
            parts.append({"text": part["text"]})
        elif kind == "image_url":
            url = part["image_url"]["url"]
            match = re.match(r"^data:([^;]+);base64,(.*)$", url, re.S)
            if not match:
                raise ValueError("Vertex transport needs base64 data URLs for images")
            parts.append({"inlineData": {"mimeType": match.group(1), "data": match.group(2)}})
        else:
            raise ValueError("Unsupported content part type: %s" % kind)
    return parts


def build_vertex_request(
    content: list[dict], *, project_id: str, model: str, temperature: float
) -> tuple[str, dict]:
    url = _VERTEX_ENDPOINT.format(project_id=project_id, model=vertex_model_name(model))
    body = {
        "contents": [{"role": "user", "parts": _vertex_parts(content)}],
        "generationConfig": {
            "temperature": temperature,
            "responseMimeType": "application/json",
        },
    }
    return url, body


def _vertex_generate(
    *,
    content: list[dict],
    model: str,
    info: dict,
    timeout: float,
    temperature: float,
) -> tuple[dict, str]:
    url, body = build_vertex_request(
        content, project_id=info["project_id"], model=model, temperature=temperature
    )
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": "Bearer %s" % vertex_access_token(info),
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError("Vertex HTTP %s: %s" % (exc.code, detail[:500])) from exc
    candidates = payload.get("candidates") or []
    if not candidates:
        raise RuntimeError("Vertex returned no candidates: %s" % json.dumps(payload)[:300])
    parts = (candidates[0].get("content") or {}).get("parts") or []
    raw = "".join(part.get("text", "") for part in parts if isinstance(part, dict))
    return _extract_json(raw), raw


def vlm_generate_json(
    content: list[dict],
    *,
    model: str = DEFAULT_VLM_MODEL,
    timeout: float = 180.0,
    temperature: float = 0.1,
    response_format: dict | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> tuple[dict, str]:
    """Send one multimodal prompt; return (parsed JSON object, raw text).

    Vertex with the service account when ``VERTEX_AI_SERVICE_ACCOUNT`` is set;
    otherwise the gateway (``GEMINI_API_KEY`` + ``BASE_URL``).
    """
    env = load_env()
    info = load_service_account(env)
    if info is not None:
        return _vertex_generate(
            content=content,
            model=model,
            info=info,
            timeout=timeout,
            temperature=temperature,
        )
    base_url = (base_url or env.get("BASE_URL") or "").rstrip("/")
    if not base_url:
        raise SystemExit(
            "VLM needs VERTEX_AI_SERVICE_ACCOUNT, or BASE_URL + GEMINI_API_KEY for the gateway"
        )
    gateway_model = _gateway_model_name(model)
    return _chat_completion(
        content=content,
        model=gateway_model,
        base_url=base_url,
        api_key=resolve_gateway_key(gateway_model, env=env, api_key=api_key),
        timeout=timeout,
        temperature=temperature,
        response_format=response_format,
    )


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
    """Parse the first JSON object from model text (tolerates trailing junk)."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    # Prefer the first complete value — models sometimes emit two objects.
    decoder = json.JSONDecoder()
    try:
        obj, _end = decoder.raw_decode(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{", text)
    if not match:
        raise json.JSONDecodeError("No JSON object found", text, 0)
    obj, _end = decoder.raw_decode(text[match.start() :])
    if not isinstance(obj, dict):
        raise ValueError("Expected a JSON object, got %s" % type(obj).__name__)
    return obj




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
    model: str = DEFAULT_VLM_MODEL,
    base_url: str | None = None,
    api_key: str | None = None,
    timeout: float = 180.0,
) -> dict:
    """Ask the VLM which catalog hypotheses are supported by the full clip.

    Prefer ``clip_path`` (animated 32-frame WebP); falls back to ``panel_path``.
    """
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

    parsed, raw = vlm_generate_json(
        content,
        model=model,
        timeout=timeout,
        temperature=0.1,
        base_url=base_url,
        api_key=api_key,
    )
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


_SOL_AUDIT_ISSUES = {
    "none",
    "too_conservative",
    "wrong_target",
    "missing_actors",
    "temporally_inconsistent",
    "graphics_or_prop",
    "other",
}

_SOL_AUDIT_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "sol_saliency_audit",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "sol_adequate": {"type": "boolean"},
                "issue": {
                    "type": "string",
                    "enum": [
                        "none",
                        "too_conservative",
                        "wrong_target",
                        "missing_actors",
                        "temporally_inconsistent",
                        "graphics_or_prop",
                        "other",
                    ],
                },
                "confidence": {"type": "number"},
                "primary_track": {"type": "string"},
                "scene_change_at": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": (
                        "frame_index values where the shot/scene focus legitimately "
                        "changes inside the window; empty if one continuous scene"
                    ),
                },
                "rationale": {"type": "string"},
                "corrected_frames": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "frame_index": {"type": "integer"},
                            "targets": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "properties": {
                                        "label": {"type": "string"},
                                        "bbox_xyxy_norm": {
                                            "type": "array",
                                            "items": {"type": "number"},
                                            "minItems": 4,
                                            "maxItems": 4,
                                        },
                                        "weight": {"type": "number"},
                                    },
                                    "required": ["label", "bbox_xyxy_norm", "weight"],
                                },
                            },
                        },
                        "required": ["frame_index", "targets"],
                    },
                },
            },
            "required": [
                "sol_adequate",
                "issue",
                "confidence",
                "primary_track",
                "scene_change_at",
                "rationale",
                "corrected_frames",
            ],
        },
    },
}


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _normalize_target(target: dict) -> dict | None:
    box = target.get("bbox_xyxy_norm")
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        x1, y1, x2, y2 = [_clamp01(v) for v in box]
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    weight = target.get("weight", 1.0)
    try:
        weight = float(weight)
    except (TypeError, ValueError):
        weight = 1.0
    return {
        "label": str(target.get("label") or target.get("target_type") or "salient").strip()[:80],
        "bbox_xyxy_norm": [round(x1, 5), round(y1, 5), round(x2, 5), round(y2, 5)],
        "weight": round(max(0.0, weight), 4),
    }


def _chat_completion(
    *,
    content: list[dict],
    model: str,
    base_url: str,
    api_key: str,
    timeout: float,
    temperature: float = 0.1,
    response_format: dict | None = None,
) -> tuple[dict, str]:
    body = {
        "model": model,
        "temperature": temperature,
        "messages": [{"role": "user", "content": content}],
    }
    if response_format is not None:
        body["response_format"] = response_format
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
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
    return _extract_json(raw), raw


def audit_sol_saliency(
    sample: dict,
    *,
    frame_images: list[Image.Image],
    frame_indices: list[int],
    sol_targets_summary: list[dict],
    model: str = DEFAULT_VLM_MODEL,
    base_url: str | None = None,
    api_key: str | None = None,
    timeout: float = 240.0,
) -> dict:
    """Ask whether Sol boxes match true scene salience; request corrections if not.

    ``frame_images`` should be RGB crops of the play (ideally with current Sol
    boxes already drawn in cyan). ``frame_indices`` are 0-based indices into the
    32-frame window. ``sol_targets_summary`` lists current Sol targets per
    provided frame for text context.
    """
    if not frame_images or len(frame_images) != len(frame_indices):
        raise ValueError("frame_images and frame_indices must be non-empty and aligned")

    prompt = {
        "task": (
            "You are auditing Sol saliency boxes for a sports broadcast clip. "
            "Each attached image is one RGB frame from a ~1s 32-frame window; "
            "cyan rectangles are the CURRENT Sol annotations used to build the "
            "ground-truth saliency map that trains a video saliency model."
        ),
        "sample": {
            "sport": sample.get("sport"),
            "game": sample.get("game"),
            "window": sample.get("window"),
            "cc": sample.get("cc"),
            "loss": sample.get("loss"),
        },
        "current_sol_boxes": sol_targets_summary,
        "goal": (
            "Produce TEMPORALLY CONSISTENT, RELEVANT visually salient target "
            "boxes for what a viewer would watch. Within a continuous shot/"
            "scene, keep a stable track (same actor/action) with boxes that "
            "move smoothly. If the scene legitimately changes inside the "
            "window (hard cut, replay wipe, camera switch, new play setup), "
            "the salient target MAY change at that boundary — do not force one "
            "track across unrelated scenes. Sol often fails by boxing only a "
            "tiny ball/logo/prop, flickering between targets inside one scene, "
            "or ignoring the players that drive the action."
        ),
        "instructions": [
            "Watch every attached frame as a short video before deciding; detect "
            "shot/scene changes.",
            "sol_adequate=true only if Sol boxes are relevant AND temporally "
            "consistent within each scene segment (no flicker to tiny secondary "
            "objects inside a continuous shot).",
            "If Sol only boxes a tiny secondary object (ball alone, graphic, prop) "
            "while a player/action is clearly more salient, set sol_adequate=false "
            "and issue=too_conservative.",
            "If Sol boxes the wrong person/region inside a scene, use issue=wrong_target.",
            "If Sol misses additional key actors that should share attention in "
            "the same scene, use issue=missing_actors.",
            "If Sol flickers between unrelated targets inside the SAME continuous "
            "scene, use issue=temporally_inconsistent. A target change that aligns "
            "with a real scene/shot change is NOT temporally_inconsistent — list "
            "those frame_index values in scene_change_at instead.",
            "When sol_adequate=false, fill corrected_frames for EVERY attached "
            "frame_index with relevant boxes. Within one scene segment use a "
            "stable label for the same actor and smooth motion. After a scene "
            "change you may switch labels/targets. Boxes are normalized xyxy in "
            "[0,1] on the attached RGB frame. Prefer torso/player/action regions "
            "large enough for saliency; ball only as optional lower-weight extra, "
            "never the sole tiny box.",
            "scene_change_at: list of frame_index where the salient scene changes "
            "(empty if one continuous scene).",
            "primary_track: short description of the salient track(s), mentioning "
            "scene changes if any.",
            "When sol_adequate=true, corrected_frames must be [] and scene_change_at "
            "may still list real cuts Sol handled correctly.",
            "Return ONLY one JSON object matching the schema (no markdown, no "
            "second object). issue must be one of %s."
            % sorted(_SOL_AUDIT_ISSUES),
        ],
    }

    content: list[dict] = [{"type": "text", "text": json.dumps(prompt, indent=2)}]
    for index, image in zip(frame_indices, frame_images):
        content.append(
            {
                "type": "text",
                "text": "Window frame_index=%d (cyan = current Sol boxes)." % index,
            }
        )
        content.append(
            {"type": "image_url", "image_url": {"url": _jpeg_data_url(image, quality=80)}}
        )

    parsed, raw = vlm_generate_json(
        content,
        model=model,
        timeout=timeout,
        temperature=0.2,
        response_format=_SOL_AUDIT_RESPONSE_FORMAT,
        base_url=base_url,
        api_key=api_key,
    )

    adequate = bool(parsed.get("sol_adequate"))
    issue = str(parsed.get("issue") or ("none" if adequate else "other")).strip()
    if issue not in _SOL_AUDIT_ISSUES:
        issue = "none" if adequate else "other"
    if adequate:
        issue = "none"
    confidence = _clamp01(parsed.get("confidence", 0.5))
    rationale = str(parsed.get("rationale") or "").strip()
    primary_track = str(parsed.get("primary_track") or "").strip()
    scene_change_at: list[int] = []
    for value in parsed.get("scene_change_at") or []:
        try:
            scene_change_at.append(int(value))
        except (TypeError, ValueError):
            continue
    scene_change_at = sorted(set(scene_change_at))

    corrected = []
    if not adequate:
        allowed = {int(i) for i in frame_indices}
        for row in parsed.get("corrected_frames") or []:
            try:
                frame_index = int(row.get("frame_index"))
            except (TypeError, ValueError, AttributeError):
                continue
            if frame_index not in allowed:
                continue
            targets = []
            for target in row.get("targets") or []:
                if not isinstance(target, dict):
                    continue
                normalized = _normalize_target(target)
                if normalized:
                    targets.append(normalized)
            if targets:
                corrected.append({"frame_index": frame_index, "targets": targets})
        corrected.sort(key=lambda item: item["frame_index"])
        # Model said inadequate but forgot boxes — still flag the issue.
        if not corrected:
            issue = issue if issue != "none" else "other"

    return {
        "sol_adequate": adequate,
        "issue": issue,
        "confidence": confidence,
        "primary_track": primary_track,
        "scene_change_at": scene_change_at,
        "rationale": rationale,
        "corrected_frames": corrected,
        "frame_indices": list(frame_indices),
        "raw": raw,
        "model": model,
    }


# --- DHF1K disagreement audit (last frame only) ---------------------------------

DHF1K_DISPOSITIONS = (
    "gt_correct_model_wrong",
    "gt_likely_wrong",
    "both_plausible_or_ambiguous",
    "insufficient_evidence",
)
DHF1K_PEAK_TARGETS = (
    "salient_subject_or_action",
    "secondary_object",
    "background",
    "graphic",
    "implausible_region",
)
SOL_COVERAGE_VALUES = ("yes", "partial", "no", "no_boxes")
SOL_LABELS = ("person", "ball", "object")
_EVIDENCE_TEXT_KEYS = ("viewer_focus", "sol_box_notes", "disagreement_cause")
_WEIGHT_SUM_TOL = 0.02
_MAX_BOX_AREA = 0.9


class VLMValidationError(ValueError):
    """Model output parsed but violates the response contract."""


def _require_unit_float(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise VLMValidationError("%s must be a number in [0, 1], got %r" % (name, value))
    value = float(value)
    if not (0.0 <= value <= 1.0):
        raise VLMValidationError("%s must be in [0, 1], got %r" % (name, value))
    return value


def validate_dhf1k_verdict(parsed: dict) -> dict:
    """Strictly validate the verdict call. Boxes are not allowed here."""
    if not isinstance(parsed, dict):
        raise VLMValidationError("verdict must be a JSON object")
    disposition = parsed.get("disposition")
    if disposition not in DHF1K_DISPOSITIONS:
        raise VLMValidationError(
            "disposition must be one of %s, got %r" % (list(DHF1K_DISPOSITIONS), disposition)
        )
    confidence = _require_unit_float(parsed.get("confidence"), "confidence")
    evidence = parsed.get("evidence")
    if not isinstance(evidence, dict):
        raise VLMValidationError("evidence must be an object")
    clean: dict = {}
    for key in _EVIDENCE_TEXT_KEYS:
        text = evidence.get(key)
        if not isinstance(text, str) or not text.strip():
            raise VLMValidationError("evidence.%s must be a non-empty string" % key)
        clean[key] = text.strip()
    coverage = evidence.get("sol_boxes_cover_subject")
    if coverage not in SOL_COVERAGE_VALUES:
        raise VLMValidationError(
            "evidence.sol_boxes_cover_subject must be one of %s, got %r"
            % (list(SOL_COVERAGE_VALUES), coverage)
        )
    peak = evidence.get("dhf1k_peak_lands_on")
    if peak not in DHF1K_PEAK_TARGETS:
        raise VLMValidationError(
            "evidence.dhf1k_peak_lands_on must be one of %s, got %r"
            % (list(DHF1K_PEAK_TARGETS), peak)
        )
    clean["sol_boxes_cover_subject"] = coverage
    clean["dhf1k_peak_lands_on"] = peak
    for forbidden in ("detections", "boxes", "bbox_xyxy_norm", "corrected_boxes"):
        if forbidden in parsed:
            raise VLMValidationError("verdict must not include %s" % forbidden)
    return {"disposition": disposition, "confidence": confidence, "evidence": clean}


def validate_sol_detections(detections) -> list[dict]:
    """Validate a complete detection list: closed labels, unit xyxy, weights sum to 1."""
    if not isinstance(detections, list):
        raise VLMValidationError("detections must be a list")
    clean = []
    for index, det in enumerate(detections):
        if not isinstance(det, dict):
            raise VLMValidationError("detections[%d] must be an object" % index)
        label = det.get("label")
        if label not in SOL_LABELS:
            raise VLMValidationError(
                "detections[%d].label must be one of %s, got %r"
                % (index, list(SOL_LABELS), label)
            )
        named = ("x_min", "y_min", "x_max", "y_max")
        if all(key in det for key in named):
            box = [det[key] for key in named]
        else:
            box = det.get("bbox_xyxy_norm")
        if not isinstance(box, list) or len(box) != 4:
            raise VLMValidationError(
                "detections[%d] needs x_min, y_min, x_max, y_max" % index
            )
        coords = [
            _require_unit_float(v, "detections[%d].bbox_xyxy_norm" % index) for v in box
        ]
        x1, y1, x2, y2 = coords
        if x2 <= x1 or y2 <= y1:
            raise VLMValidationError(
                "detections[%d].bbox_xyxy_norm needs x_max>x_min and y_max>y_min" % index
            )
        if (x2 - x1) * (y2 - y1) >= _MAX_BOX_AREA:
            raise VLMValidationError(
                "detections[%d] covers the whole frame; boxes must be tight" % index
            )
        weight = det.get("weight")
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or weight <= 0:
            raise VLMValidationError("detections[%d].weight must be a positive number" % index)
        confidence = _require_unit_float(det.get("confidence"), "detections[%d].confidence" % index)
        clean.append(
            {
                "label": label,
                "bbox_xyxy_norm": [round(v, 5) for v in coords],
                "weight": float(weight),
                "confidence": confidence,
            }
        )
    if clean:
        total = sum(det["weight"] for det in clean)
        if abs(total - 1.0) > _WEIGHT_SUM_TOL:
            raise VLMValidationError("detection weights must sum to 1.0, got %.4f" % total)
    return clean


def validate_sol_correction(parsed: dict) -> dict:
    if not isinstance(parsed, dict):
        raise VLMValidationError("correction must be a JSON object")
    issues = parsed.get("issues_found")
    if not isinstance(issues, bool):
        raise VLMValidationError("issues_found must be a boolean")
    notes = parsed.get("notes")
    if not isinstance(notes, str) or not notes.strip():
        raise VLMValidationError("notes must be a non-empty string")
    detections = validate_sol_detections(parsed.get("detections"))
    if issues and not detections:
        raise VLMValidationError(
            "issues_found=true requires the complete corrected detections list"
        )
    return {"issues_found": issues, "notes": notes.strip(), "detections": detections}


def _generate_validated(
    content: list[dict],
    validator,
    *,
    transport,
    model: str,
    timeout: float,
    temperature: float,
) -> tuple[dict, list[str]]:
    """Call once; on parse/contract failure retry once with the validator error."""
    raws: list[str] = []
    error: Exception | None = None
    for attempt in range(2):
        attempt_content = list(content)
        if error is not None:
            attempt_content.append(
                {
                    "type": "text",
                    "text": (
                        "Your previous response was invalid: %s\n"
                        "Return one corrected JSON object only." % error
                    ),
                }
            )
        try:
            parsed, raw = transport(
                attempt_content, model=model, timeout=timeout, temperature=temperature
            )
        except (json.JSONDecodeError, ValueError) as exc:
            if isinstance(exc, VLMValidationError):
                raise
            error = VLMValidationError("response was not a JSON object (%s)" % exc)
            raws.append("")
            continue
        raws.append(raw)
        try:
            return validator(parsed), raws
        except VLMValidationError as exc:
            error = exc
    raise VLMValidationError(str(error)) from error


def _default_transport(content, *, model, timeout, temperature):
    return vlm_generate_json(content, model=model, timeout=timeout, temperature=temperature)


def _image_part(image: Image.Image, quality: int = 85) -> dict:
    return {"type": "image_url", "image_url": {"url": _jpeg_data_url(image, quality=quality)}}


def _disagreement_context(sample: dict) -> dict:
    comps = sample.get("components") or {}
    return {
        key: (round(float(comps[key]), 4) if comps.get(key) is not None else None)
        for key in ("kldiv", "cc", "sim", "peak_distance_norm", "shape_mismatch")
    }


def _sol_box_listing(targets: list[dict]) -> list[dict]:
    rows = []
    for index, target in enumerate(targets or []):
        rows.append(
            {
                "index": index,
                "label": target.get("label") or target.get("target_type") or "",
                "bbox_xyxy_norm": target.get("bbox_xyxy_norm"),
                "weight": target.get("weight"),
            }
        )
    return rows


def build_dhf1k_verdict_content(
    sample: dict,
    *,
    sol_image: Image.Image,
    gt_overlay: Image.Image,
    pred_overlay: Image.Image,
    sol_targets: list[dict],
) -> list[dict]:
    """Verdict prompt: last frame only, three images, no box drawing requested."""
    prompt = {
        "task": (
            "You are adjudicating one frame from a %s broadcast. A baseline "
            "saliency model (DHF1K) and the current ground-truth annotation "
            "strongly disagree on this frame. Decide, from the frame itself, "
            "which side is more plausible." % (sample.get("sport") or "sports")
        ),
        "images": [
            "Image 1: the RGB frame with the current Sol annotation boxes drawn "
            "(yellow=person, cyan=ball, magenta=object; '#index label' printed).",
            "Image 2: the ground-truth saliency map built from those boxes, as a "
            "jet heat overlay (red = most salient).",
            "Image 3: the DHF1K baseline prediction, as a jet heat overlay.",
        ],
        "current_sol_boxes": _sol_box_listing(sol_targets),
        "disagreement_metrics": _disagreement_context(sample),
        "questions": [
            "What would a typical viewer watch on this frame, and why?",
            "Do the current Sol boxes cover that subject or action with an "
            "appropriate spatial extent?",
            "Does the DHF1K peak land on a salient subject or action, a secondary "
            "object, background, a graphic, or an implausible region?",
            "Is the disagreement model error, a bad annotation, or genuine ambiguity?",
        ],
        "instructions": [
            "Answer the first question from the RGB frame alone before looking at "
            "either heat map. Neither map is assumed correct.",
            "Judge only this frame. You are not shown other frames.",
            "disposition=gt_correct_model_wrong when the annotation covers what a "
            "viewer watches and the model peak does not.",
            "disposition=gt_likely_wrong when the annotation misses, mislabels, "
            "or badly fits what a viewer watches.",
            "disposition=both_plausible_or_ambiguous when both regions are "
            "reasonable viewer targets or the frame has no clear focus.",
            "disposition=insufficient_evidence when the frame is too dark, "
            "blurred, occluded, or graphics-dominated to judge.",
            "Do not draw or return boxes.",
            "Return ONLY one JSON object: {\"disposition\": one of %s, "
            "\"confidence\": number in [0,1], \"evidence\": {\"viewer_focus\": "
            "string, \"sol_boxes_cover_subject\": one of %s, \"sol_box_notes\": "
            "string, \"dhf1k_peak_lands_on\": one of %s, \"disagreement_cause\": "
            "string}}."
            % (
                list(DHF1K_DISPOSITIONS),
                list(SOL_COVERAGE_VALUES),
                list(DHF1K_PEAK_TARGETS),
            ),
        ],
    }
    return [
        {"type": "text", "text": json.dumps(prompt, indent=2)},
        {"type": "text", "text": "Image 1: frame with current Sol boxes."},
        _image_part(sol_image),
        {"type": "text", "text": "Image 2: ground-truth saliency overlay."},
        _image_part(gt_overlay),
        {"type": "text", "text": "Image 3: DHF1K prediction overlay."},
        _image_part(pred_overlay),
    ]


def build_sol_correction_content(
    sample: dict,
    *,
    sol_image: Image.Image,
    sol_targets: list[dict],
) -> list[dict]:
    """Check-and-fix prompt on the frame with labeled boxes already drawn."""
    width, height = sol_image.size
    sport = sample.get("sport") or "sports"
    text = "\n".join(
        [
            "This is one frame from a %s broadcast, %d x %d pixels. The current "
            "Sol saliency annotations are drawn on it: yellow=person, cyan=ball, "
            "magenta=object, each with '#index label' printed next to the box."
            % (sport, width, height),
            "The frame was flagged because a baseline saliency model and these "
            "annotations strongly disagree. That flag does not say which is right; "
            "judge the annotations against the frame itself.",
            "",
            "Current boxes (normalized xyxy):",
            json.dumps(_sol_box_listing(sol_targets)),
            "",
            "For every drawn box, check:",
            "1. Is it the right class?",
            "2. Is it in the right place?",
            "3. Is it tightly fit around the actual subject (not loose, not whole "
            "frame, not background)?",
            "Also check whether the frame has a confident, salient instance of a "
            "primary target that has no box drawn at all.",
            "",
            "Classes:",
            "- person: competing athlete or key official driving viewer attention. "
            "Tight around the full visible body (head to feet), or head/shoulders "
            "for close-ups. Do not label crowd, coaches, bench, commentators, or "
            "cameramen.",
            "- ball: the active ball in play, only when clearly visible and driving "
            "focus. Tight box around just the ball. Never use a tiny ball as the "
            "sole target if a player is what commands attention.",
            "- object: compact non-person focal item (e.g. hoop, goal net, racquet, "
            "trophy). Tight box around that item, not the arm, court, or background.",
            "",
            "Decision:",
            "- If every drawn box is correct and nothing confident is missing: "
            "issues_found=false, notes=\"Current annotations are accurate\", and "
            "return the existing boxes unchanged in detections.",
            "- If anything is wrong (wrong target, box too loose or shifted, a box "
            "that should not exist, or a missing salient subject): issues_found=true, "
            "a concise one-sentence explanation in notes, and the COMPLETE corrected "
            "list of detections for this frame. Never return only the changes.",
            "",
            "Coordinates and rules:",
            "- bbox_xyxy_norm is [x_min, y_min, x_max, y_max] floats in [0, 1] on "
            "this image, top-left origin. Never return pixel values or 0-1000 integers.",
            "- Each detection: label (person|ball|object), bbox_xyxy_norm, weight "
            "(weights sum to 1.0), confidence in [0, 1].",
            "- Tight boxes only. Never box the whole court/field or the whole frame.",
            "- Judge this image on its own.",
            "",
            "Return ONLY one JSON object: {\"issues_found\": bool, \"notes\": string, "
            "\"detections\": [{\"label\": string, \"bbox_xyxy_norm\": [x_min, y_min, "
            "x_max, y_max], \"weight\": number, \"confidence\": number}]}.",
        ]
    )
    return [{"type": "text", "text": text}, _image_part(sol_image, quality=90)]


def adjudicate_dhf1k_disagreement(
    sample: dict,
    *,
    sol_image: Image.Image,
    gt_overlay: Image.Image,
    pred_overlay: Image.Image,
    sol_targets: list[dict],
    model: str = DEFAULT_VLM_MODEL,
    timeout: float = 180.0,
    transport=None,
) -> dict:
    """Verdict-only call. Returns status ok|invalid|error with disposition when ok."""
    content = build_dhf1k_verdict_content(
        sample,
        sol_image=sol_image,
        gt_overlay=gt_overlay,
        pred_overlay=pred_overlay,
        sol_targets=sol_targets,
    )
    try:
        verdict, raws = _generate_validated(
            content,
            validate_dhf1k_verdict,
            transport=transport or _default_transport,
            model=model,
            timeout=timeout,
            temperature=0.1,
        )
    except VLMValidationError as exc:
        return {"status": "invalid", "error": str(exc), "model": model}
    except Exception as exc:
        return {"status": "error", "error": str(exc), "model": model}
    return {"status": "ok", **verdict, "attempts": len(raws), "raw": raws[-1], "model": model}


def build_directed_box_content(image: Image.Image, instruction: str, *, sport: str) -> list[dict]:
    """Ask for boxes for the salient thing a reviewer named, on one clean frame."""
    width, height = image.size
    text = "\n".join(
        [
            "You are given 1 frame from a live %s broadcast. Image size is %d x %d pixels."
            % (sport or "sports", width, height),
            "",
            "A reviewer rejected both the current ground-truth boxes and any previously "
            "proposed boxes. They say what should be marked as salient on this frame:",
            instruction.strip(),
            "",
            "Draw boxes only for what the reviewer described. Do not box anything else.",
            "",
            "Each box uses four named fields, not an array: x_min, y_min, x_max, y_max. "
            "They are floats in [0, 1] on THIS image. Origin is the top-left, x increases "
            "right, y increases down. x_min is the left edge, y_min is the top edge. "
            "Do not return [y_min, x_min, y_max, x_max]. Do not return pixel values. "
            "Do not return 0-1000 coordinates.",
            "",
            "label must be exactly one of: person, ball, object.",
            "- person: tight box around the full visible body, head to feet, or around "
            "the head and shoulders on a close-up. A standing player is taller than wide. "
            "Not the court around them.",
            "- ball: tight box around the ball only, not the player near it.",
            "- object: tight box around that object only, not the arm or the stand behind it.",
            "",
            "Usually one box. A second box only when the reviewer described more than one "
            "target. weight values are in (0, 1] and sum to 1. confidence is your certainty, "
            "from 0 to 1.",
            "Tight boxes only. Never box the whole court or the whole frame.",
            "Judge this image on its own. Return detections for this image.",
            "",
            "Return ONLY one JSON object: {\"notes\": one short sentence, \"detections\": "
            "[{\"label\": string, \"x_min\": number, \"y_min\": number, \"x_max\": number, "
            "\"y_max\": number, \"weight\": number, \"confidence\": number}]}.",
        ]
    )
    return [{"type": "text", "text": text}, _image_part(image, quality=90)]


def validate_directed_boxes(parsed: dict) -> dict:
    if not isinstance(parsed, dict):
        raise VLMValidationError("directed boxes must be a JSON object")
    notes = parsed.get("notes")
    if not isinstance(notes, str) or not notes.strip():
        raise VLMValidationError("notes must be a non-empty string")
    detections = upright_person_boxes(validate_sol_detections(parsed.get("detections")))
    if not detections:
        raise VLMValidationError("detections must list the boxes for what the reviewer described")
    return {"notes": notes.strip(), "detections": detections}


def upright_person_boxes(detections: list[dict]) -> list[dict]:
    """Turn a flat person box into a tall one when the axes were swapped.

    Gemini often returns ``[y_min, x_min, y_max, x_max]``. Read as xyxy, a
    standing player becomes a horizontal strip. Swapping the axes restores
    the tall box when that swap is itself taller than it is wide.
    """
    fixed = []
    for det in detections:
        box = [float(v) for v in det["bbox_xyxy_norm"]]
        width = box[2] - box[0]
        height = box[3] - box[1]
        if det.get("label") == "person" and width > 0 and height < width * 0.45:
            swapped = [box[1], box[0], box[3], box[2]]
            swapped_w = swapped[2] - swapped[0]
            swapped_h = swapped[3] - swapped[1]
            if swapped_w > 0 and swapped_h > swapped_w:
                box = swapped
        fixed.append({**det, "bbox_xyxy_norm": [round(v, 5) for v in box]})
    return fixed


def direct_sol_boxes(
    image: Image.Image,
    instruction: str,
    *,
    sport: str = "sports",
    model: str = DEFAULT_VLM_MODEL,
    timeout: float = 180.0,
    transport=None,
) -> dict:
    """Boxes for a reviewer-named salient target. status valid|invalid|error."""
    instruction = (instruction or "").strip()
    if not instruction:
        return {"status": "invalid", "error": "prompt is empty", "detections": [], "model": model}
    content = build_directed_box_content(image, instruction, sport=sport)
    try:
        result, raws = _generate_validated(
            content,
            validate_directed_boxes,
            transport=transport or _default_transport,
            model=model,
            timeout=timeout,
            temperature=0.1,
        )
    except VLMValidationError as exc:
        return {"status": "invalid", "error": str(exc), "detections": [], "model": model}
    except Exception as exc:
        return {"status": "error", "error": str(exc), "detections": [], "model": model}
    return {
        "status": "valid",
        **result,
        "attempts": len(raws),
        "model": model,
    }


def correct_sol_boxes(
    sample: dict,
    *,
    sol_image: Image.Image,
    sol_targets: list[dict],
    model: str = DEFAULT_VLM_MODEL,
    timeout: float = 180.0,
    transport=None,
) -> dict:
    """Box-correction call (only for gt_likely_wrong).

    status: ``valid`` (issues_found with a corrected list), ``no_issues``,
    ``invalid`` (contract failed after one retry; boxes left uncorrected), or
    ``error`` (transport failure).
    """
    content = build_sol_correction_content(sample, sol_image=sol_image, sol_targets=sol_targets)
    try:
        result, raws = _generate_validated(
            content,
            validate_sol_correction,
            transport=transport or _default_transport,
            model=model,
            timeout=timeout,
            temperature=0.1,
        )
    except VLMValidationError as exc:
        return {"status": "invalid", "error": str(exc), "detections": [], "model": model}
    except Exception as exc:
        return {"status": "error", "error": str(exc), "detections": [], "model": model}
    status = "valid" if result["issues_found"] else "no_issues"
    return {
        "status": status,
        **result,
        "attempts": len(raws),
        "raw": raws[-1],
        "model": model,
    }
