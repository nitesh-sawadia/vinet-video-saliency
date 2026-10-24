"""Serve the review UI from an analysis output directory."""

from __future__ import annotations

import functools
import http.server
import os
import shutil
import socketserver
import webbrowser


def serve_review(out_dir: str, host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True):
    out_dir = os.path.abspath(out_dir)
    index_path = os.path.join(out_dir, "review_index.json")
    if not os.path.isfile(index_path):
        raise SystemExit("Missing review_index.json in %s; run classify first" % out_dir)

    ui_src = os.path.join(os.path.dirname(__file__), "review_ui.html")
    ui_dst = os.path.join(out_dir, "index.html")
    shutil.copyfile(ui_src, ui_dst)

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=out_dir)
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer((host, port), handler) as httpd:
        url = "http://%s:%d/" % (host, port)
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
