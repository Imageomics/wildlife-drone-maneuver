"""Stage 3: cut the 6-second mp4 clips and write per-frame-per-track labels.

For each selected clip: seek the source video to `start_frame`, write
`CLIP_FRAMES` frames to `clips/<clip_id>/clip.mp4` (cv2, mp4v), and write
`clips/<clip_id>/labels.csv` -- one row per frame per track over the window with
KABR-native fields (bbox geometry, species, behaviour, vigilance, bbox size
class, telemetry). `pose`/`individual_id` columns are created empty here and
filled by stage 4 / left for future work.

Usage:
    python -m clip_library.extract_clips [--out DIR] [--limit N] [--clips id,id]
"""

from __future__ import annotations

import os
import argparse

import numpy as np
import pandas as pd

from . import schema, io_paths


def _build_labels(occ: pd.DataFrame, clip: pd.Series, w: int, h: int) -> pd.DataFrame:
    s, e = int(clip["start_frame"]), int(clip["end_frame"])
    t = occ[occ["xtl"].notna()].copy()
    for c in ("xtl", "ytl", "xbr", "ybr", "frame"):
        t[c] = pd.to_numeric(t[c], errors="coerce")
    t = t.dropna(subset=["xtl", "ytl", "xbr", "ybr", "frame"])
    t["frame"] = t["frame"].astype(int)
    t = t[(t["frame"] >= s) & (t["frame"] < e)]
    # occurrence files carry ~10x exact-duplicate rows per (frame, track) from the
    # detection/behaviour merge -> collapse to one row per frame per track.
    t = t.drop_duplicates(subset=["frame", "id", "xtl", "ytl", "xbr", "ybr", "behaviour"])

    rows = []
    for r in t.itertuples():
        wpx = max(0.0, r.xbr - r.xtl)
        hpx = max(0.0, r.ybr - r.ytl)
        area_frac = (wpx * hpx) / float(w * h) if (w and h) else 0.0
        beh = schema.normalize_behaviour(None if pd.isna(r.behaviour) else r.behaviour)
        rows.append({
            "clip_id": clip["clip_id"],
            "video_id": clip["video_id"],
            "session_id": clip["session_id"],
            "frame_global": int(r.frame),
            "frame_local": int(r.frame) - s,
            "time_s": round((int(r.frame) - s) / schema.FPS, 3),
            "track_id": r.id,
            "species": schema.normalize_species(None if pd.isna(r.label) else r.label),
            "behaviour": beh,
            "vigilant": beh in schema.VIGILANCE_BEHAVIOURS,
            "pose": "",
            "pose_provenance": "",
            "pose_match_score": "",
            "individual_id": "",   # future work: no global re-ID GT exists
            "xtl": round(float(r.xtl), 2),
            "ytl": round(float(r.ytl), 2),
            "xbr": round(float(r.xbr), 2),
            "ybr": round(float(r.ybr), 2),
            "x_c": round(float(r.xtl + wpx / 2), 2),
            "y_c": round(float(r.ytl + hpx / 2), 2),
            "w": round(wpx, 2),
            "h": round(hpx, 2),
            "bbox_area_frac": round(area_frac, 6),
            "bbox_size_class": schema.bbox_size_class(area_frac),
            "occluded": _truthy(getattr(r, "occluded_x", None)),
            "outside": _truthy(getattr(r, "outside_x", None)),
            "latitude": getattr(r, "latitude", None),
            "longitude": getattr(r, "longitude", None),
            "altitude": getattr(r, "altitude", None),
            "date_time": "" if pd.isna(getattr(r, "date_time", None)) else str(r.date_time),
        })
    df = pd.DataFrame(rows, columns=schema.LABEL_COLUMNS)
    return df.sort_values(["frame_local", "track_id"]).reset_index(drop=True)


def _truthy(v) -> bool:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return False
    return str(v).strip() not in ("", "0", "0.0", "False", "false", "nan")


def _cut_video(src: str, start: int, n: int, dst: str) -> tuple[int, int, int]:
    """Write `n` frames from `start` to dst (mp4v). Returns (written, w, h)."""
    import cv2

    cap = cv2.VideoCapture(src)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or schema.FPS
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(dst, fourcc, fps, (w, h))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    written = 0
    try:
        for _ in range(n):
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
            written += 1
    finally:
        writer.release()
        cap.release()
    return written, w, h


def main():
    ap = argparse.ArgumentParser(description="Stage 3: extract clips + labels")
    ap.add_argument("--out", default=schema.OUTPUT_ROOT)
    ap.add_argument("--limit", type=int, default=None, help="only first N clips")
    ap.add_argument("--clips", default=None, help="comma-separated clip_ids to (re)extract")
    ap.add_argument("--labels-only", action="store_true", help="skip mp4 writing")
    args = ap.parse_args()

    cindex = io_paths.read_table(os.path.join(args.out, "catalog", "clip_index"))
    if args.clips:
        wanted = set(args.clips.split(","))
        cindex = cindex[cindex["clip_id"].isin(wanted)]
    if args.limit:
        cindex = cindex.head(args.limit)

    # group by source video / occurrence so we open each once
    cindex = cindex.sort_values(["video_id", "start_frame"])
    occ_cache: dict[str, pd.DataFrame] = {}
    n_ok = n_fail = 0
    for _, clip in cindex.iterrows():
        src = clip["source_video_path"]
        clip_dir = os.path.join(args.out, "clips", clip["clip_id"])
        if not src or not os.path.exists(src):
            print(f"  SKIP {clip['clip_id']}: source video missing ({src})")
            n_fail += 1
            continue

        # occurrence rows (cache by path)
        occ_path = None
        # recover occurrence path from clip_id's date_prefix+video
        occ_path = os.path.join(
            schema.OCCURRENCE_DIR, f"{clip['date_prefix']}-{clip['video_id']}.csv"
        )
        if occ_path not in occ_cache:
            occ_cache[occ_path] = pd.read_csv(occ_path, low_memory=False)
        occ = occ_cache[occ_path]

        # cut video
        if args.labels_only:
            w, h, _, _ = io_paths.video_dims(src)
            written = schema.CLIP_FRAMES
        else:
            written, w, h = _cut_video(
                src, int(clip["start_frame"]), schema.CLIP_FRAMES,
                os.path.join(clip_dir, "clip.mp4"),
            )

        labels = _build_labels(occ, clip, w, h)
        io_paths.write_table(labels, os.path.join(clip_dir, "labels"))

        status = "ok" if written == schema.CLIP_FRAMES else f"short({written})"
        print(f"  {clip['clip_id']:<34s} frames={written:<3d} tracks_rows={len(labels):<4d} {status}")
        n_ok += 1 if written == schema.CLIP_FRAMES else 0

    print(f"\nextracted {n_ok} clips ok, {n_fail} skipped (of {len(cindex)})")


if __name__ == "__main__":
    main()
