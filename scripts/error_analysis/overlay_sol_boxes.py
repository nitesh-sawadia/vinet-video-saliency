"""Re-bake Sol boxes into existing hard-example clips/panels without re-inference.

Replaces the left third of each rendered panel (frame) with frame + Sol boxes
loaded from the window ``boxes.json`` / Linode ``pre_split`` fallback, keeping
the existing GT and prediction overlays.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from viz import load_rgb_u8, load_sol_native_targets, draw_sol_boxes_rgb


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _default_pre_split_root(split_root: str | None) -> str | None:
    if not split_root:
        return None
    parent = os.path.dirname(os.path.abspath(split_root.rstrip(os.sep)))
    candidate = os.path.join(parent, "pre_split")
    return candidate if os.path.isdir(candidate) else None


def _manifest_by_rel(split_root: str) -> dict[str, dict]:
    path = Path(split_root) / "split_manifest.json"
    if not path.is_file():
        return {}
    manifest = json.loads(path.read_text())
    by_rel: dict[str, dict] = {}
    for window in manifest.get("windows") or []:
        if not isinstance(window, dict):
            continue
        name = Path(str(window.get("dir") or "")).name
        rel = f'{window.get("split")}/{window.get("sport")}/{window.get("game")}/{name}'
        by_rel[rel] = window
    return by_rel


def _replace_left(panel: np.ndarray, left: np.ndarray) -> np.ndarray:
    height, width = panel.shape[:2]
    third = width // 3
    if left.shape[0] != height or left.shape[1] != third:
        left = np.asarray(
            Image.fromarray(left).resize((third, height), Image.BILINEAR)
        )
    out = panel.copy()
    out[:, :third] = left
    return out


def overlay_sol_boxes(
    out_dir: str,
    split_root: str | None = None,
    pre_split_root: str | None = None,
    ids: list[str] | None = None,
    fps: int = 8,
) -> dict:
    out = Path(out_dir)
    scores_path = out / "scores.jsonl"
    if not scores_path.is_file():
        raise SystemExit(f"Missing scores.jsonl in {out_dir}")
    rows = _read_jsonl(scores_path)
    hard = [row for row in rows if row.get("hard")]
    if ids:
        want = set(ids)
        hard = [row for row in hard if str(row.get("id")) in want]

    summary_path = out / "summary.json"
    if split_root is None and summary_path.is_file():
        split_root = json.loads(summary_path.read_text()).get("split_root")
    if not split_root:
        raise SystemExit("--split-root is required")
    if pre_split_root is None:
        pre_split_root = _default_pre_split_root(split_root)
    man_by_rel = _manifest_by_rel(split_root)

    print(
        f"Overlaying Sol boxes on {len(hard)} hard clips "
        f"(pre_split={pre_split_root or 'none'}) ...",
        flush=True,
    )
    updated = 0
    with_boxes = 0
    for row in hard:
        sample_id = str(row["id"])
        window_path = Path(row.get("path") or (Path(split_root) / (row.get("relpath") or "")))
        man = man_by_rel.get(str(row.get("relpath") or "")) or {}
        targets = load_sol_native_targets(
            window_path,
            pre_split_root=pre_split_root,
            source=man.get("source"),
            native_start=man.get("native_start"),
            native_end=man.get("native_end"),
        )
        row["sol_boxes"] = bool(targets)
        if targets:
            with_boxes += 1

        clip_rel = row.get("clip") or f"clips/{sample_id}.webp"
        clip_path = out / clip_rel
        if clip_path.is_file() and window_path.is_dir():
            webp = Image.open(clip_path)
            n_frames = getattr(webp, "n_frames", 1)
            panels = []
            for t in range(n_frames):
                webp.seek(t)
                panel = np.asarray(webp.convert("RGB"))
                frame_path = window_path / "images" / f"{t + 1:04d}.png"
                if frame_path.is_file():
                    left = load_rgb_u8(str(frame_path))
                    if t < len(targets):
                        left = draw_sol_boxes_rgb(left, targets[t])
                    panel = _replace_left(panel, left)
                panels.append(Image.fromarray(panel))
            duration_ms = max(1, int(round(1000.0 / float(fps))))
            panels[0].save(
                clip_path,
                format="WEBP",
                save_all=True,
                append_images=panels[1:],
                duration=duration_ms,
                loop=0,
                quality=80,
                method=4,
            )
            updated += 1

        panel_rel = row.get("panel") or f"panels/{sample_id}.png"
        panel_path = out / panel_rel
        if panel_path.is_file() and window_path.is_dir() and targets:
            panel = np.asarray(Image.open(panel_path).convert("RGB"))
            last = load_rgb_u8(str(window_path / "images" / "0032.png"))
            last = draw_sol_boxes_rgb(last, targets[-1])
            Image.fromarray(_replace_left(panel, last)).save(panel_path)

        print(
            f"  {sample_id} boxes={len(targets)} clip={'ok' if clip_path.is_file() else 'missing'}",
            flush=True,
        )

    _write_jsonl(scores_path, rows)
    sol_by_id = {row["id"]: bool(row.get("sol_boxes")) for row in rows if row.get("hard")}
    for name in ("review_index.json", "analysis.json"):
        path = out / name
        if not path.is_file():
            continue
        doc = json.loads(path.read_text())
        for sample in doc.get("samples") or []:
            sid = sample.get("id")
            if sid in sol_by_id:
                sample["sol_boxes"] = sol_by_id[sid]
                sample["media_version"] = "sol-boxes"
        path.write_text(json.dumps(doc, indent=2) + "\n")
        print("Updated", path, flush=True)

    result = {"n_updated": updated, "n_with_sol_boxes": with_boxes, "n_hard": len(hard)}
    print(json.dumps(result), flush=True)
    return result


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split-root", default=None)
    parser.add_argument("--pre-split-root", default=None)
    parser.add_argument("--ids", default=None, help="Comma-separated hard ids")
    parser.add_argument("--fps", type=int, default=8)
    args = parser.parse_args(argv)
    ids = None
    if args.ids:
        ids = [item.strip() for item in args.ids.split(",") if item.strip()]
    overlay_sol_boxes(
        args.out,
        split_root=args.split_root,
        pre_split_root=args.pre_split_root,
        ids=ids,
        fps=args.fps,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
