"""Stage 5: render bbox+label overlays and contact sheets for manual review.

The user signs off the label join (raw video -> bbox -> species/behaviour/pose/
telemetry) by eye. This stage draws every track box on a stratified sample of
clips with a caption `track | species | behaviour | pose`, writing both an
overlay mp4 and a 3x3 contact sheet per clip, plus `qa/manifest.csv` (the
sign-off sheet, with `verified` / `notes` columns to fill in).

Usage:
    python -m clip_library.make_qa_overlays [--out DIR] [--stratified] [--all]
                                            [--clips id,id]
"""

from __future__ import annotations

import os
import argparse
import collections

import numpy as np
import pandas as pd

from . import schema, io_paths

# distinct BGR colors per species token (fallbacks cycle)
_PALETTE = [
    (66, 133, 244), (52, 168, 83), (251, 188, 5), (234, 67, 53),
    (171, 71, 188), (0, 172, 193), (255, 112, 67), (158, 157, 36),
]


def _species_color(species: str, table: dict) -> tuple:
    if species not in table:
        table[species] = _PALETTE[len(table) % len(_PALETTE)]
    return table[species]


def _draw(frame, rows, color_table):
    import cv2

    for r in rows.itertuples():
        x1, y1, x2, y2 = int(r.xtl), int(r.ytl), int(r.xbr), int(r.ybr)
        col = _species_color(str(r.species), color_table)
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
        pose = "" if (pd.isna(r.pose) or r.pose == "") else f" | {r.pose}"
        cap = f"{r.track_id} | {r.species} | {r.behaviour}{pose}"
        y_text = y1 - 6 if y1 > 16 else y2 + 16
        cv2.putText(frame, cap, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, col, 1, cv2.LINE_AA)
    return frame


def render_clip(clip_id, out_root):
    import cv2

    clip_dir = os.path.join(out_root, "clips", clip_id)
    mp4 = os.path.join(clip_dir, "clip.mp4")
    lab_path = os.path.join(clip_dir, "labels")
    if not (os.path.exists(mp4) and os.path.exists(lab_path + ".csv")):
        return False
    labels = io_paths.read_table(lab_path)
    qa_dir = os.path.join(out_root, "qa")
    os.makedirs(qa_dir, exist_ok=True)
    color_table: dict = {}

    cap = cv2.VideoCapture(mp4)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or schema.FPS
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(
        os.path.join(qa_dir, f"{clip_id}_overlay.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    sheet_idxs = set(np.linspace(0, max(n - 1, 0), 9).astype(int))
    sheet_tiles = {}

    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rows = labels[labels["frame_local"] == fi]
        frame = _draw(frame, rows, color_table)
        writer.write(frame)
        if fi in sheet_idxs:
            sheet_tiles[fi] = cv2.resize(frame, (w // 3, h // 3))
        fi += 1
    writer.release()
    cap.release()

    # 3x3 contact sheet
    tiles = [sheet_tiles[k] for k in sorted(sheet_tiles)]
    if tiles:
        th, tw = tiles[0].shape[:2]
        while len(tiles) < 9:
            tiles.append(np.zeros((th, tw, 3), np.uint8))
        grid = np.vstack([np.hstack(tiles[r * 3:(r + 1) * 3]) for r in range(3)])
        cv2.imwrite(os.path.join(qa_dir, f"{clip_id}_contact_sheet.jpg"), grid)
    return True


def stratified_sample(cindex: pd.DataFrame) -> list[str]:
    """Pick clips spanning session, species, habitat, size-class, maneuver, + all pose videos."""
    chosen = collections.OrderedDict()

    def add(cid):
        chosen[cid] = True

    # all 5 pose videos (best pose coverage each)
    pose = cindex[cindex["pose_frames_covered"] > 0]
    for vid, g in pose.groupby("video_id"):
        add(g.sort_values("pose_frames_covered", ascending=False).iloc[0]["clip_id"])

    covered = collections.defaultdict(set)
    for cid in chosen:
        _mark(cindex[cindex.clip_id == cid].iloc[0], covered)

    # greedily add to fill each axis value
    for _, r in cindex.iterrows():
        if r["clip_id"] in chosen:
            continue
        new = _mark(r, covered, dry_run=True)
        if new:
            add(r["clip_id"])
            _mark(r, covered)
    return list(chosen.keys())


def _mark(r, covered, dry_run=False):
    axes = {
        "session": [str(r.get("session_id", ""))],
        "habitat": [str(r.get("habitat", ""))],
        "species": [s for s in str(r.get("species_set", "")).split("|") if s],
        "size": [s for s in str(r.get("bbox_size_classes", "")).split("|") if s],
        "maneuver": [s for s in str(r.get("suitable_maneuvers", "")).split("|") if s],
    }
    new = False
    for ax, vals in axes.items():
        for v in vals:
            if v and v not in covered[ax]:
                new = True
                if not dry_run:
                    covered[ax].add(v)
    return new


def main():
    ap = argparse.ArgumentParser(description="Stage 5: QA overlays + manifest")
    ap.add_argument("--out", default=schema.OUTPUT_ROOT)
    ap.add_argument("--stratified", action="store_true",
                    help="render a stratified review sample (default if no selector)")
    ap.add_argument("--all", action="store_true", help="render every clip")
    ap.add_argument("--clips", default=None, help="comma-separated clip_ids")
    args = ap.parse_args()

    cindex = io_paths.read_table(os.path.join(args.out, "catalog", "clip_index"))

    if args.clips:
        ids = [c for c in args.clips.split(",")]
    elif args.all:
        ids = cindex["clip_id"].tolist()
    else:
        ids = stratified_sample(cindex)
    print(f"Rendering QA overlays for {len(ids)} clips ...")

    done = []
    for cid in ids:
        ok = render_clip(cid, args.out)
        print(f"  {'ok ' if ok else 'SKIP'} {cid}")
        if ok:
            done.append(cid)

    # manifest = the sign-off sheet
    sel = cindex[cindex["clip_id"].isin(done)].copy()
    keep = ["clip_id", "video_id", "session_id", "habitat", "species_set",
            "bbox_size_classes", "suitable_maneuvers", "pose_frames_covered",
            "pose_set", "start_frame", "source_video_path"]
    keep = [c for c in keep if c in sel.columns]
    man = sel[keep].copy()
    man["verified"] = ""
    man["notes"] = ""
    man_path = os.path.join(args.out, "qa", "manifest")
    io_paths.write_table(man, man_path)
    print(f"\nwrote {len(done)} overlays + contact sheets to {os.path.join(args.out,'qa')}")
    print(f"sign-off sheet: {man_path}.csv  (fill 'verified'/'notes' per clip)")


if __name__ == "__main__":
    main()
