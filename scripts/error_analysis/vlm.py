"""Vision-LLM confirmation of failure hypotheses via the Amagi LLM gateway."""

from __future__ import annotations

import base64
import json
import os
import re
import urllib.error
import urllib.request

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


def _encode_image(path: str) -> str:
    with open(path, "rb") as handle:
        return base64.b64encode(handle.read()).decode("ascii")


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


def confirm_hypotheses(
    panel_path: str,
    sample: dict,
    proposals: list[dict],
    *,
    model: str = "vertex_ai/gemini-2.5-flash",
    base_url: str | None = None,
    api_key: str | None = None,
    timeout: float = 90.0,
) -> dict:
    """Ask the VLM which catalog hypotheses are supported by the panel.

    Returns {
      confirmed: [{id, confidence, rationale}],
      rejected_proposals: [...],
      raw: str,
      model: str,
    }
    """
    env = load_env()
    base_url = (base_url or env.get("BASE_URL") or "").rstrip("/")
    api_key = api_key or env.get("OPENAI_API_KEY")
    if not base_url or not api_key:
        raise SystemExit(
            "VLM needs BASE_URL and OPENAI_API_KEY in the environment or repo .env"
        )
    if not os.path.isfile(panel_path):
        raise FileNotFoundError(panel_path)

    catalog = [
        {"id": hid, "title": HYPOTHESIS_BY_ID[hid]["title"], "description": HYPOTHESIS_BY_ID[hid]["description"]}
        for hid in HYPOTHESIS_IDS
        if hid != "unexplained"
    ]
    proposed_ids = [p["id"] for p in proposals]
    prompt = {
        "task": (
            "You are reviewing a sports video-saliency failure. The image is a "
            "1x3 panel: left=last RGB frame, middle=ground-truth saliency overlay, "
            "right=model prediction overlay."
        ),
        "sample": {
            "sport": sample.get("sport"),
            "game": sample.get("game"),
            "window": sample.get("window"),
            "cc": sample.get("cc"),
            "kldiv": sample.get("kldiv"),
            "loss": sample.get("loss"),
            "proposed_hypothesis_ids": proposed_ids,
            "proposal_evidence": proposals,
        },
        "catalog": catalog,
        "instructions": [
            "Confirm only hypotheses clearly supported by the panel and metrics.",
            "You may confirm catalog ids that were not proposed if visually obvious.",
            "Prefer 1-2 confirmed ids; use none if nothing fits.",
            "Do not invent ids outside the catalog.",
            "Return strict JSON with keys confirmed (list of {id, confidence 0-1, rationale}) "
            "and notes (short string).",
        ],
    }

    body = {
        "model": model,
        "temperature": 0.1,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": json.dumps(prompt, indent=2)},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,%s" % _encode_image(panel_path)
                        },
                    },
                ],
            }
        ],
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
    }
