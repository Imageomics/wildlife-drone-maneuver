#!/usr/bin/env python3
"""Decision-support: render side-by-side overlays of two parallel KABR annotation
passes (e.g. 16_01_23_flight_1 vs flight_2) on the SAME raw frames, so a human can
judge which pass is the higher-quality / canonical release.

Both passes annotate the same source video (verified: their occurrence CSVs embed
the source SRT telemetry, which matches `part2`). This tool reads a stratified set
of frames from that raw video, draws pass-A boxes on the left panel and pass-B
boxes on the right, captions each box `id | label | behaviour`, and writes one PNG
per sampled frame plus a small index.

Usage:
    python -m clip_library.compare_flights \
        --video DJI_0002 --a 16_01_23_flight_1 --b 16_01_23_flight_2 \
        --raw .../part2/DJI_0002.MP4 --n 6
"""
from __future__ import annotations

import os
import csv
import argparse
import collections

import cv2

from . import schema

# Mode-B QA utility (raw-video diagnostics); paths point at data not shipped
# with this repo. Override via env or --flags. `schema.OCCURRENCE_DIR` derives
# from KABR_ROOT; the others default under it / OUTPUT_ROOT.
OCC_DIR = schema.OCCURRENCE_DIR
RAW_PART2 = os.environ.get("COMPARE_RAW_PART2", "")
OUT_DEFAULT = os.path.join(schema.OUTPUT_ROOT, "qa", "flight_compare")

A_COLOR = (60, 180, 75)    # green  (BGR)
B_COLOR = (60, 76, 231)    # red


def load_boxes(occ_csv: str) -> dict:
    """frame -> list of (id, xtl, ytl, xbr, ybr, label, behaviour) for visible boxes."""
    by_frame = collections.defaultdict(list)
    with open(occ_csv, errors="ignore") as f:
        for r in csv.DictReader(f):
            try:
                fr = int(r["frame"])
                x1, y1, x2, y2 = (float(r["xtl"]), float(r["ytl"]),
                                  float(r["xbr"]), float(r["ybr"]))
            except (ValueError, KeyError, TypeError):
                continue
            if r.get("outside_x") == "1" or r.get("outside_y") == "1":
                continue
            by_frame[fr].append((r.get("id", ""), x1, y1, x2, y2,
                                 r.get("label", ""), r.get("behaviour", "")))
    return by_frame


def draw_panel(frame, boxes, color, title):
    img = frame.copy()
    for (tid, x1, y1, x2, y2, label, behav) in boxes:
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(img, p1, p2, color, 3)
        cap = f"{tid}|{behav}" if behav else f"{tid}|{label}"
        cv2.putText(img, cap, (int(x1), max(0, int(y1) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
    cv2.rectangle(img, (0, 0), (img.shape[1], 70), (0, 0, 0), -1)
    cv2.putText(img, f"{title}  ({len(boxes)} boxes)", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="e.g. DJI_0002")
    ap.add_argument("--a", default="16_01_23_flight_1")
    ap.add_argument("--b", default="16_01_23_flight_2")
    ap.add_argument("--raw", default=None, help="raw mp4; defaults to part2/<video>.MP4")
    ap.add_argument("--n", type=int, default=6, help="frames to sample")
    ap.add_argument("--out", default=OUT_DEFAULT)
    args = ap.parse_args()

    raw = args.raw or f"{RAW_PART2}/{args.video}.MP4"
    a_csv = f"{OCC_DIR}/{args.a}-{args.video}.csv"
    b_csv = f"{OCC_DIR}/{args.b}-{args.video}.csv"
    out_dir = os.path.join(args.out, args.video)
    os.makedirs(out_dir, exist_ok=True)

    a_boxes = load_boxes(a_csv)
    b_boxes = load_boxes(b_csv)

    cap = cv2.VideoCapture(raw)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {raw}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # sample frames where BOTH passes have boxes, spread across the span
    common = sorted(set(a_boxes) & set(b_boxes))
    if not common:
        raise SystemExit("no frames where both passes have visible boxes")
    step = max(1, len(common) // args.n)
    sampled = common[::step][: args.n]

    print(f"{args.video}: A={args.a}({len(a_boxes)} fr)  B={args.b}({len(b_boxes)} fr)  "
          f"raw={total} fr  -> sampling {sampled}")

    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    for fr in sampled:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
        ok, frame = cap.read()
        if not ok:
            print(f"  frame {fr}: read failed"); continue
        left = draw_panel(frame, a_boxes.get(fr, []), A_COLOR, f"{args.a}")
        right = draw_panel(frame, b_boxes.get(fr, []), B_COLOR, f"{args.b}")
        divider = cv2.copyMakeBorder(left, 0, 0, 0, 8, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        combo = cv2.hconcat([divider, right])
        # downscale for quick viewing
        combo = cv2.resize(combo, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_AREA)
        path = os.path.join(out_dir, f"frame_{fr:06d}.png")
        cv2.imwrite(path, combo)
        print(f"  wrote {path}  (A={len(a_boxes.get(fr,[]))} vs B={len(b_boxes.get(fr,[]))} boxes)")
    cap.release()


if __name__ == "__main__":
    main()
