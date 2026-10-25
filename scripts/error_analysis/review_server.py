"""Serve the review UI from an analysis output directory.

For DHF1K audits (``review_index.json`` with ``kind == "dhf1k_audit"``) the
server also records decisions:

  GET  /api/reviews  -> human_reviews.json (``{}`` when none yet)
  POST /api/review   {"id": ..., "decision": "accept"|"reject"|"skip", "note": ...}

Decisions are written to ``human_reviews.json`` in the audit directory, keyed
by sample id; a new decision overwrites that sample's record. Accepted boxes
are copied from the server-side review index, never from the request body.
Nothing outside the audit directory is written.
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

REVIEW_DECISIONS = ("accept", "reject", "skip")
_MAX_BODY_BYTES = 64 * 1024


class _ReviewStore:
    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        self.path = os.path.join(out_dir, "human_reviews.json")
        self.lock = threading.Lock()

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
            correction = sample.get("correction") or {}
            if not sample.get("accept_allowed") or not correction.get("detections"):
                raise ValueError(
                    "accept needs verdict gt_likely_wrong and a valid box correction"
                )
            record["accepted_boxes"] = correction["detections"]
            record["frame_index"] = 31
        with self.lock:
            reviews = self.read()
            reviews[sample_id] = record
            fd, tmp = tempfile.mkstemp(prefix=".human_reviews.", dir=self.out_dir)
            with os.fdopen(fd, "w") as handle:
                json.dump(reviews, handle, indent=2, sort_keys=True)
                handle.write("\n")
            os.replace(tmp, self.path)
        return record


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
        if self.path.split("?", 1)[0] != "/api/review":
            self._send_json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0 or length > _MAX_BODY_BYTES:
                raise ValueError("bad request body size")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("body must be a JSON object")
            record = self.store.record(payload)
        except (ValueError, json.JSONDecodeError) as exc:
            self._send_json(400, {"error": str(exc)})
            return
        self._send_json(200, record)


def make_review_server(out_dir: str, host: str = "127.0.0.1", port: int = 8765):
    """Build (but do not start) the review HTTP server. ``port=0`` picks a free port."""
    out_dir = os.path.abspath(out_dir)
    index_path = os.path.join(out_dir, "review_index.json")
    if not os.path.isfile(index_path):
        raise SystemExit("Missing review_index.json in %s; run classify first" % out_dir)

    ui_src = os.path.join(os.path.dirname(__file__), "review_ui.html")
    ui_dst = os.path.join(out_dir, "index.html")
    shutil.copyfile(ui_src, ui_dst)

    handler_cls = type("ReviewHandler", (_ReviewHandler,), {"store": _ReviewStore(out_dir)})
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
