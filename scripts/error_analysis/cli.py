#!/usr/bin/env python3
"""Outplay validation error analysis CLI.

Examples:
  python -m error_analysis score \\
    --checkpoint /path/to/best.pt \\
    --split-root /path/to/split \\
    --out /tmp/outplay_ea

  python -m error_analysis classify --out /tmp/outplay_ea
  python -m error_analysis review --out /tmp/outplay_ea

  python -m error_analysis run \\
    --checkpoint ... --split-root ... --out ... --review
"""

from __future__ import annotations

import argparse
import os
import sys


def _bootstrap():
    here = os.path.dirname(os.path.abspath(__file__))
    scripts = os.path.dirname(here)
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    if here not in sys.path:
        sys.path.insert(0, here)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    score = sub.add_parser("score", help="Score all val windows and write hard-example panels")
    score.add_argument("--checkpoint", required=True, help="best.pt training bundle or weights-only .pt")
    score.add_argument("--split-root", required=True, help="Outplay split root with split_manifest.json")
    score.add_argument("--out", required=True, help="Output directory")
    score.add_argument("--batch-size", type=int, default=16,
                        help="clips per forward pass (default 16)")
    score.add_argument("--num-workers", type=int, default=4)
    score.add_argument("--device", default=None, help="cuda|cpu (default: auto)")
    score.add_argument("--hard-fraction", type=float, default=0.12)
    score.add_argument("--hard-min", type=int, default=40)
    score.add_argument("--hard-max", type=int, default=80)

    classify = sub.add_parser("classify", help="Propose + VLM-confirm hypotheses for hard examples")
    classify.add_argument("--out", required=True, help="Directory produced by score")
    classify.add_argument("--vlm-model", default="vertex_ai/gemini-2.5-flash")
    classify.add_argument("--skip-vlm", action="store_true", help="Use geometric proposals only")
    classify.add_argument("--limit", type=int, default=None, help="Optional cap on hard examples")

    review = sub.add_parser("review", help="Serve the bucketed review UI")
    review.add_argument("--out", required=True)
    review.add_argument("--host", default="127.0.0.1")
    review.add_argument("--port", type=int, default=8765)
    review.add_argument("--no-browser", action="store_true")

    render = sub.add_parser(
        "render-clips",
        help="Render 32-frame comparison videos for hard examples",
    )
    render.add_argument("--out", required=True, help="Directory produced by score")
    render.add_argument("--checkpoint", required=True)
    render.add_argument("--split-root", default=None, help="Defaults to summary.json split_root")
    render.add_argument("--batch-size", type=int, default=8)
    render.add_argument("--num-workers", type=int, default=2)
    render.add_argument("--device", default=None)
    render.add_argument("--fps", type=int, default=8)

    run = sub.add_parser("run", help="score + classify (+ optional review)")
    run.add_argument("--checkpoint", required=True)
    run.add_argument("--split-root", required=True)
    run.add_argument("--out", required=True)
    run.add_argument("--batch-size", type=int, default=16,
                     help="clips per forward pass (default 16)")
    run.add_argument("--num-workers", type=int, default=4)
    run.add_argument("--device", default=None)
    run.add_argument("--hard-fraction", type=float, default=0.12)
    run.add_argument("--hard-min", type=int, default=40)
    run.add_argument("--hard-max", type=int, default=80)
    run.add_argument("--vlm-model", default="vertex_ai/gemini-2.5-flash")
    run.add_argument("--skip-vlm", action="store_true")
    run.add_argument("--limit", type=int, default=None)
    run.add_argument("--review", action="store_true")
    run.add_argument("--host", default="127.0.0.1")
    run.add_argument("--port", type=int, default=8765)
    run.add_argument("--no-browser", action="store_true")
    return parser


def main(argv=None) -> int:
    # Line-buffer stdout/stderr when redirected to a log file.
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass
    _bootstrap()
    args = build_parser().parse_args(argv)

    if args.command == "score":
        from score import score_validation

        score_validation(
            checkpoint_path=args.checkpoint,
            split_root=args.split_root,
            out_dir=args.out,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device_name=args.device,
            hard_fraction=args.hard_fraction,
            hard_min=args.hard_min,
            hard_max=args.hard_max,
        )
        return 0

    if args.command == "classify":
        from classify import classify_hard_examples

        classify_hard_examples(
            args.out,
            vlm_model=args.vlm_model,
            skip_vlm=args.skip_vlm,
            limit=args.limit,
        )
        return 0

    if args.command == "review":
        from review_server import serve_review

        serve_review(
            args.out,
            host=args.host,
            port=args.port,
            open_browser=not args.no_browser,
        )
        return 0

    if args.command == "render-clips":
        from render_clips import render_hard_clips

        render_hard_clips(
            out_dir=args.out,
            checkpoint_path=args.checkpoint,
            split_root=args.split_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device_name=args.device,
            fps=args.fps,
        )
        return 0

    if args.command == "run":
        from score import score_validation
        from classify import classify_hard_examples

        score_validation(
            checkpoint_path=args.checkpoint,
            split_root=args.split_root,
            out_dir=args.out,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device_name=args.device,
            hard_fraction=args.hard_fraction,
            hard_min=args.hard_min,
            hard_max=args.hard_max,
        )
        classify_hard_examples(
            args.out,
            vlm_model=args.vlm_model,
            skip_vlm=args.skip_vlm,
            limit=args.limit,
        )
        if args.review:
            from review_server import serve_review

            serve_review(
                args.out,
                host=args.host,
                port=args.port,
                open_browser=not args.no_browser,
            )
        return 0

    raise SystemExit("unknown command")


if __name__ == "__main__":
    raise SystemExit(main())
