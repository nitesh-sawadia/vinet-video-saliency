"""Serve the review UI from an analysis output directory.

For DHF1K audits (``review_index.json`` with ``kind == "dhf1k_audit"``) the
server also records decisions:

  GET  /api/reviews  -> human_reviews.json (``{}`` when none yet)
  POST /api/review   {"id": ..., "decision": "accept"|"reject"|"skip", "note": ...}
  POST /api/direct-correction  {"id": ..., "prompt": ...}

Decisions are written to ``human_reviews.json`` in the audit directory, keyed
by sample id; a new decision overwrites that sample's record. Accepted boxes
are copied from the server-side review index, never from the request body.
A direct-correction prompt asks the VLM to box what the reviewer described
and stores those boxes on the sample. Nothing outside the audit directory
is written.
"""

from __future__ import annotations

import functools
import http.server
import json
import os
import shutil
import socketserver
import tempfile
import threading
import time
import webbrowser

import numpy as np
from PIL import Image

from viz import draw_indexed_boxes_rgb
from vlm import DEFAULT_VLM_MODEL, direct_sol_boxes

REVIEW_DECISIONS = ("accept", "reject", "skip")
_MAX_BODY_BYTES = 64 * 1024


class _ReviewStore:
    def __init__(self, out_dir: str, box_fn=None):
        self.out_dir = out_dir
        self.path = os.path.join(out_dir, "human_reviews.json")
        self.index_path = os.path.join(out_dir, "review_index.json")
        self.directed_path = os.path.join(out_dir, "directed_corrections.json")
        self.lock = threading.Lock()
        self.box_fn = box_fn or _call_directed_boxes

    def _samples(self) -> dict:
        with open(os.path.join(self.out_dir, "review_index.json")) as handle:
            index = json.load(handle)
        return {str(s["id"]): s for s in index.get("samples") or []}

    def read(self) -> dict:
        if not os.path.isfile(self.path):
            return {}
        with open(self.path) as handle:
            return json.load(handle)

    def record(self, payload: dict) -> dict:
        sample_id = str(payload.get("id") or "")
        decision = payload.get("decision")
        if decision not in REVIEW_DECISIONS:
            raise ValueError("decision must be one of %s" % list(REVIEW_DECISIONS))
        sample = self._samples().get(sample_id)
        if sample is None:
            raise ValueError("unknown sample id %r" % sample_id)
        record = {
            "id": sample_id,
            "decision": decision,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "relpath": sample.get("relpath"),
            "sport": sample.get("sport"),
            "game": sample.get("game"),
            "window": sample.get("window"),
            "disposition": (sample.get("verdict") or {}).get("disposition"),
        }
        note = payload.get("note")
        if isinstance(note, str) and note.strip():
            record["note"] = note.strip()[:2000]
        if decision == "accept":
            source = payload.get("box_source") or ""
            directed = sample.get("directed_correction") or {}
            correction = sample.get("correction") or {}
            prompted = directed.get("status") == "valid" and directed.get("detections")
            audited = bool(correction.get("status") == "valid" and correction.get("detections"))
            if source == "directed" or (not source and prompted):
                if not prompted:
                    raise ValueError("no prompted boxes to accept as ground truth")
                record["accepted_boxes"] = directed["detections"]
                record["box_source"] = "directed"
                record["prompt"] = directed.get("prompt")
            elif source == "vlm" or not source:
                if not audited:
                    raise ValueError(
                        "accept needs a valid box list from the audit or from a directed prompt"
                    )
                record["accepted_boxes"] = correction["detections"]
                record["box_source"] = "vlm"
            else:
                raise ValueError("box_source must be directed or vlm")
            record["frame_index"] = 31
            record["gt_map"] = "maps/0032.png"
        with self.lock:
            reviews = self.read()
            reviews[sample_id] = record
            fd, tmp = tempfile.mkstemp(prefix=".human_reviews.", dir=self.out_dir)
            with os.fdopen(fd, "w") as handle:
                json.dump(reviews, handle, indent=2, sort_keys=True)
                handle.write("\n")
            os.replace(tmp, self.path)
        return record

    def direct(self, payload: dict) -> dict:
        """Ask the VLM to box the salient target the reviewer described."""
        sample_id = str(payload.get("id") or "")
        prompt = payload.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must describe what should be marked as salient")
        prompt = prompt.strip()[:2000]
        with self.lock:
            index = self._read_index()
            samples = index.get("samples") or []
            sample = next((s for s in samples if str(s.get("id")) == sample_id), None)
            if sample is None:
                raise ValueError("unknown sample id %r" % sample_id)
            frame_rel = (sample.get("media") or {}).get("frame")
            if not frame_rel:
                raise ValueError("sample has no last-frame image")
            frame_path = os.path.join(self.out_dir, frame_rel)
            image = Image.open(frame_path).convert("RGB")
            result = self.box_fn(image, prompt, sample)
            result = {
                "status": result.get("status") or "error",
                "notes": result.get("notes") or "",
                "detections": result.get("detections") or [],
                "error": result.get("error") or "",
                "model": result.get("model") or DEFAULT_VLM_MODEL,
                "prompt": prompt,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            media_rel = ""
            if result["status"] == "valid" and result["detections"]:
                media_rel = "media/%s_directed.png" % sample_id
                drawn = draw_indexed_boxes_rgb(np.asarray(image), result["detections"])
                os.makedirs(os.path.join(self.out_dir, "media"), exist_ok=True)
                Image.fromarray(drawn).save(os.path.join(self.out_dir, media_rel))
                sample["directed_media"] = media_rel
                sample["accept_allowed"] = True
            sample["directed_correction"] = {
                key: result[key]
                for key in ("status", "notes", "detections", "error", "model", "prompt", "timestamp")
            }
            self._write_index(index)
            directed = self._read_json(self.directed_path)
            directed[sample_id] = sample["directed_correction"]
            if media_rel:
                directed[sample_id]["media"] = media_rel
            self._write_json(self.directed_path, directed)
        return {
            "id": sample_id,
            "status": result["status"],
            "notes": result["notes"],
            "detections": result["detections"],
            "error": result["error"],
            "directed_media": media_rel,
            "accept_allowed": bool(sample.get("accept_allowed")),
            "prompt": prompt,
        }

    def _read_index(self) -> dict:
        with open(self.index_path) as handle:
            return json.load(handle)

    def _write_index(self, index: dict) -> None:
        self._write_json(self.index_path, index)

    def _read_json(self, path: str) -> dict:
        if not os.path.isfile(path):
            return {}
        with open(path) as handle:
            return json.load(handle)

    def _write_json(self, path: str, payload: dict) -> None:
        fd, tmp = tempfile.mkstemp(prefix=".review.", dir=self.out_dir)
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        os.replace(tmp, path)


class _ReviewHandler(http.server.SimpleHTTPRequestHandler):
    store: _ReviewStore | None = None

    def _send_json(self, status: int, payload) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.split("?", 1)[0] == "/api/reviews":
            self._send_json(200, self.store.read())
            return
        super().do_GET()

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path not in ("/api/review", "/api/direct-correction"):
            self._send_json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0 or length > _MAX_BODY_BYTES:
                raise ValueError("bad request body size")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("body must be a JSON object")
            if path == "/api/direct-correction":
                record = self.store.direct(payload)
            else:
                record = self.store.record(payload)
        except (ValueError, json.JSONDecodeError, OSError) as exc:
            self._send_json(400, {"error": str(exc)})
            return
        self._send_json(200, record)


def _call_directed_boxes(image, prompt: str, sample: dict) -> dict:
    return direct_sol_boxes(
        image,
        prompt,
        sport=str(sample.get("sport") or "sports"),
        model=DEFAULT_VLM_MODEL,
    )


def make_review_server(out_dir: str, host: str = "127.0.0.1", port: int = 8765, box_fn=None):
    """Build (but do not start) the review HTTP server. ``port=0`` picks a free port."""
    out_dir = os.path.abspath(out_dir)
    index_path = os.path.join(out_dir, "review_index.json")
    if not os.path.isfile(index_path):
        raise SystemExit("Missing review_index.json in %s; run classify first" % out_dir)

    ui_src = os.path.join(os.path.dirname(__file__), "review_ui.html")
    ui_dst = os.path.join(out_dir, "index.html")
    shutil.copyfile(ui_src, ui_dst)

    handler_cls = type(
        "ReviewHandler", (_ReviewHandler,), {"store": _ReviewStore(out_dir, box_fn=box_fn)}
    )
    handler = functools.partial(handler_cls, directory=out_dir)
    socketserver.TCPServer.allow_reuse_address = True
    return socketserver.ThreadingTCPServer((host, port), handler)


def serve_review(out_dir: str, host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True):
    out_dir = os.path.abspath(out_dir)
    with make_review_server(out_dir, host=host, port=port) as httpd:
        url = "http://%s:%d/" % (host, httpd.server_address[1])
        print("Review UI at %s  (serving %s)" % (url, out_dir), flush=True)
        if open_browser:
            try:
                webbrowser.open(url)
            except Exception:
                pass
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.", flush=True)
