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

  # DHF1K disagreement audit (split_stride8 val, last frame only)
  python -m error_analysis audit-dhf1k \\
    --checkpoint /path/to/vinet_s_dhf1k.pt \\
    --split-root /mnt/ml-fanpulse/sports_saliency/vinet_outplay/split_stride8 \\
    --out /tmp/dhf1k_audit --stages score,select
  python -m error_analysis audit-dhf1k --out /tmp/dhf1k_audit --stages adjudicate,index
  python -m error_analysis review --out /tmp/dhf1k_audit
  python -m error_analysis export-accepted --out /tmp/dhf1k_audit
"""

from __future__ import annotations

import argparse
import os
import sys

_DEFAULT_VLM_MODEL = "gemini-3.6-flash"
_AUDIT_STAGES = ("score", "select", "adjudicate", "index")


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
    classify.add_argument("--vlm-model", default=_DEFAULT_VLM_MODEL)
    classify.add_argument("--skip-vlm", action="store_true", help="Use geometric proposals only")
    classify.add_argument("--limit", type=int, default=None, help="Optional cap on hard examples")

    review = sub.add_parser("review", help="Serve the bucketed review UI")
    review.add_argument("--out", required=True)
    review.add_argument("--host", default="127.0.0.1")
    review.add_argument("--port", type=int, default=8765)
    review.add_argument("--no-browser", action="store_true")

    audit = sub.add_parser(
        "audit-sol",
        help="VLM-check whether Sol boxes match scene salience; request corrections",
    )
    audit.add_argument("--out", required=True, help="Directory produced by score/classify")
    audit.add_argument("--split-root", default=None, help="Defaults to summary.json split_root")
    audit.add_argument(
        "--pre-split-root",
        default=None,
        help="Linode pre_split root for Sol boxes fallback",
    )
    audit.add_argument("--vlm-model", default=_DEFAULT_VLM_MODEL)
    audit.add_argument(
        "--max-frames",
        type=int,
        default=8,
        help="Keyframe budget sent to the VLM (default 8 of 32)",
    )
    audit.add_argument("--limit", type=int, default=None)
    audit.add_argument(
        "--ids",
        default=None,
        help="Comma-separated hard-example ids (default: all hard)",
    )
    audit.add_argument(
        "--no-resume",
        action="store_true",
        help="Re-audit samples even if present in sol_audit.jsonl",
    )
    audit.add_argument(
        "--rebuild-media-only",
        action="store_true",
        help="Rebuild 32-frame corrected WebP tracks from sol_audit.jsonl (no VLM)",
    )

    render = sub.add_parser(
        "render-clips",
        help="Render 32-frame comparison videos for hard examples",
    )
    render.add_argument("--out", required=True, help="Directory produced by score")
    render.add_argument("--checkpoint", required=True)
    render.add_argument("--split-root", default=None, help="Defaults to summary.json split_root")
    render.add_argument(
        "--pre-split-root",
        default=None,
        help="Linode pre_split root for Sol boxes fallback (default: sibling of split-root)",
    )
    render.add_argument("--batch-size", type=int, default=8)
    render.add_argument("--num-workers", type=int, default=2)
    render.add_argument("--device", default=None)
    render.add_argument("--fps", type=int, default=8)
    render.add_argument(
        "--ids",
        default=None,
        help="Comma-separated hard-example ids to re-render (default: all hard)",
    )
    render.add_argument(
        "--refresh-panels",
        action="store_true",
        help="Overwrite last-frame panels as well as 32-frame clips",
    )

    dhf = sub.add_parser(
        "audit-dhf1k",
        help="Rank DHF1K-vs-GT last-frame disagreements on split_stride8 val; VLM adjudicate",
    )
    dhf.add_argument("--out", required=True, help="Audit output dir (outside the dataset tree)")
    dhf.add_argument(
        "--stages",
        default=",".join(_AUDIT_STAGES),
        help="Comma list of %s (default: all)" % ",".join(_AUDIT_STAGES),
    )
    dhf.add_argument("--checkpoint", default=None, help="vinet_s_dhf1k.pt (score stage)")
    dhf.add_argument(
        "--split-root",
        default=None,
        help="split_stride8 root with split_manifest.json (score/select; read-only)",
    )
    dhf.add_argument("--pre-split-root", default=None, help="pre_split root for Sol boxes fallback")
    dhf.add_argument("--batch-size", type=int, default=16)
    dhf.add_argument("--num-workers", type=int, default=4)
    dhf.add_argument("--device", default=None, help="cuda|cpu (default: auto)")
    dhf.add_argument(
        "--top-percent",
        type=float,
        default=5.0,
        help="Candidate percentile cap: top N%% of eligible windows (default 5)",
    )
    dhf.add_argument("--min-candidates", type=int, default=50)
    dhf.add_argument("--max-candidates", type=int, default=400)
    dhf.add_argument("--vlm-model", default=_DEFAULT_VLM_MODEL)
    dhf.add_argument("--limit", type=int, default=None, help="Cap on candidates sent to the VLM")
    dhf.add_argument("--ids", default=None, help="Comma-separated candidate ids for adjudicate")
    dhf.add_argument(
        "--no-resume",
        action="store_true",
        help="Rescore / reselect / re-adjudicate instead of resuming prior outputs",
    )
    dhf.add_argument(
        "--allow-other-split",
        action="store_true",
        help="Permit a split root not named split_stride8 (fixtures only)",
    )

    export = sub.add_parser(
        "export-accepted",
        help="Write accepted_corrections.jsonl from human_reviews.json (decision=accept only)",
    )
    export.add_argument("--out", required=True, help="Audit dir with human_reviews.json")

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
    run.add_argument("--vlm-model", default=_DEFAULT_VLM_MODEL)
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

    if args.command == "audit-sol":
        from audit_sol import audit_hard_sol_boxes, rebuild_corrected_media

        if args.rebuild_media_only:
            rebuild_corrected_media(args.out)
            return 0

        ids = None
        if args.ids:
            ids = [item.strip() for item in args.ids.split(",") if item.strip()]
        audit_hard_sol_boxes(
            args.out,
            split_root=args.split_root,
            pre_split_root=args.pre_split_root,
            vlm_model=args.vlm_model,
            max_frames=args.max_frames,
            limit=args.limit,
            ids=ids,
            resume=not args.no_resume,
        )
        return 0

    if args.command == "audit-dhf1k":
        from audit_dhf1k import run_audit_dhf1k

        stages = [item.strip() for item in args.stages.split(",") if item.strip()]
        unknown = [stage for stage in stages if stage not in _AUDIT_STAGES]
        if unknown or not stages:
            raise SystemExit("--stages must be a subset of %s" % ",".join(_AUDIT_STAGES))
        ids = None
        if args.ids:
            ids = [item.strip() for item in args.ids.split(",") if item.strip()]
        run_audit_dhf1k(
            args.out,
            stages=stages,
            checkpoint_path=args.checkpoint,
            split_root=args.split_root,
            pre_split_root=args.pre_split_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device_name=args.device,
            top_percent=args.top_percent,
            min_candidates=args.min_candidates,
            max_candidates=args.max_candidates,
            vlm_model=args.vlm_model,
            limit=args.limit,
            ids=ids,
            resume=not args.no_resume,
            allow_other_split=args.allow_other_split,
        )
        return 0

    if args.command == "export-accepted":
        from audit_dhf1k import export_accepted

        export_accepted(args.out)
        return 0

    if args.command == "render-clips":
        from render_clips import render_hard_clips

        ids = None
        if args.ids:
            ids = [item.strip() for item in args.ids.split(",") if item.strip()]
        render_hard_clips(
            out_dir=args.out,
            checkpoint_path=args.checkpoint,
            split_root=args.split_root,
            pre_split_root=args.pre_split_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device_name=args.device,
            fps=args.fps,
            ids=ids,
            refresh_panels=args.refresh_panels,
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
