"""Stage 4: cross-reference ground-truth pose labels onto clip tracks.

The DINOv2 pose classifier was trained on KABR crops, so we do NOT run it here
(that would be inference on training data). Instead we attach the *manually
labeled* KABR-poses crops to the matching KABR track.

A crop's filename gives an exact (video, frame, class_id, det_index) but the
det_index does NOT map to a CVAT/KABR track id. So at each labeled frame we take
the candidate track boxes (from the clip labels) and pick the one whose pixels
best match the labeled crop image (visual disambiguation). Only ~5 videos carry
GT pose; everything else stays `pose = null` by design.

Usage:
    python -m clip_library.add_pose_labels [--out DIR] [--ambig-thresh F] [--audit]
"""

from __future__ import annotations

import os
import argparse
import collections

import numpy as np
import pandas as pd

from . import schema, io_paths, pose_gt

CMP_SIZE = 224           # compare crops at this square size (matches KABR-poses)
DEFAULT_AMBIG = 0.30     # TM_CCOEFF_NORMED below this => 'gt-ambiguous'


def _square_crop(frame, x1, y1, x2, y2):
    """Crop bbox from a BGR frame, grayscale, resize to CMP_SIZE^2."""
    import cv2

    h, w = frame.shape[:2]
    x1 = max(0, int(round(x1))); y1 = max(0, int(round(y1)))
    x2 = min(w, int(round(x2))); y2 = min(h, int(round(y2)))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    chip = frame[y1:y2, x1:x2]
    chip = cv2.cvtColor(chip, cv2.COLOR_BGR2GRAY)
    return cv2.resize(chip, (CMP_SIZE, CMP_SIZE))


def _similarity(a, b) -> float:
    """Normalized correlation between two equal-size gray images."""
    import cv2

    if a is None or b is None:
        return -1.0
    res = cv2.matchTemplate(a, b, cv2.TM_CCOEFF_NORMED)
    return float(res[0, 0])


def _assign_frame(frame, crops_here, cand_rows, ambig_thresh):
    """Greedily assign each labeled crop to its best-matching candidate track.

    crops_here: list of (pose_label, gray224 crop image)
    cand_rows : DataFrame of candidate track rows at this frame (one per track)
    returns dict track_id -> (pose_label, score, provenance)
    """
    cand = []
    for r in cand_rows.itertuples():
        g = _square_crop(frame, r.xtl, r.ytl, r.xbr, r.ybr)
        cand.append((r.track_id, g))

    pairs = []  # (score, crop_idx, cand_idx)
    for ci, (_, cimg) in enumerate(crops_here):
        for kj, (_, gimg) in enumerate(cand):
            pairs.append((_similarity(cimg, gimg), ci, kj))
    pairs.sort(reverse=True)

    used_crop, used_cand = set(), set()
    out = {}
    for score, ci, kj in pairs:
        if ci in used_crop or kj in used_cand:
            continue
        used_crop.add(ci); used_cand.add(kj)
        tid = cand[kj][0]
        pose_label = crops_here[ci][0]
        prov = "gt" if score >= ambig_thresh else "gt-ambiguous"
        out[tid] = (pose_label, round(score, 3), prov)
    return out


def process_clip(clip, pose_rows, out_root, ambig_thresh, audit):
    import cv2

    clip_dir = os.path.join(out_root, "clips", clip["clip_id"])
    lab_path = os.path.join(clip_dir, "labels")
    if not os.path.exists(lab_path + ".csv"):
        return None
    labels = io_paths.read_table(lab_path)
    # ensure object/numeric dtypes so string pose assignment is clean
    for col in ("pose", "pose_provenance"):
        labels[col] = labels[col].fillna("").astype(object)
    labels["pose_match_score"] = pd.to_numeric(labels["pose_match_score"], errors="coerce")
    mp4 = os.path.join(clip_dir, "clip.mp4")
    if not os.path.exists(mp4):
        return None

    start = int(clip["start_frame"])
    # pose crops within this window, keyed by frame
    by_frame = collections.defaultdict(list)
    for r in pose_rows.itertuples():
        if start <= r.frame < start + schema.CLIP_FRAMES:
            by_frame[int(r.frame)].append((r.pose_label, r.crop_path))

    if not by_frame:
        return None

    cap = cv2.VideoCapture(mp4)
    assigned = 0
    ambiguous = 0
    scores = []
    poses_in_clip = set()
    for fg, items in sorted(by_frame.items()):
        flocal = fg - start
        cap.set(cv2.CAP_PROP_POS_FRAMES, flocal)
        ok, frame = cap.read()
        if not ok:
            continue
        crops_here = [(lbl, cv2.resize(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2GRAY),
                                       (CMP_SIZE, CMP_SIZE)))
                      for lbl, p in items if os.path.exists(p)]
        crops_here = [c for c in crops_here if c[1] is not None]
        cand_rows = labels[labels["frame_global"] == fg]
        if cand_rows.empty or not crops_here:
            continue
        result = _assign_frame(frame, crops_here, cand_rows, ambig_thresh)
        for tid, (pose_label, score, prov) in result.items():
            mask = (labels["frame_global"] == fg) & (labels["track_id"] == tid)
            labels.loc[mask, "pose"] = pose_label
            labels.loc[mask, "pose_provenance"] = prov
            labels.loc[mask, "pose_match_score"] = score
            assigned += 1
            ambiguous += int(prov == "gt-ambiguous")
            scores.append(score)
            poses_in_clip.add(pose_label)
    cap.release()

    io_paths.write_table(labels, lab_path)
    return {
        "clip_id": clip["clip_id"],
        "video_id": clip["video_id"],
        "n_crops": sum(len(v) for v in by_frame.values()),
        "assigned": assigned,
        "ambiguous": ambiguous,
        "median_score": round(float(np.median(scores)), 3) if scores else None,
        "pose_set": "|".join(sorted(poses_in_clip)),
    }


def main():
    ap = argparse.ArgumentParser(description="Stage 4: GT pose cross-reference")
    ap.add_argument("--out", default=schema.OUTPUT_ROOT)
    ap.add_argument("--ambig-thresh", type=float, default=DEFAULT_AMBIG)
    ap.add_argument("--clips", default=None, help="comma-separated clip_ids")
    args = ap.parse_args()

    cindex = io_paths.read_table(os.path.join(args.out, "catalog", "clip_index"))
    pose_df = pose_gt.load_pose_crops()

    targets = cindex[cindex["pose_frames_covered"] > 0]
    if args.clips:
        targets = targets[targets["clip_id"].isin(set(args.clips.split(",")))]
    print(f"Cross-referencing GT pose on {len(targets)} pose-bearing clips ...")

    audits = []
    pose_set_by_clip = {}
    for _, clip in targets.iterrows():
        # pose rows for this clip's exact (video, date/session)
        pr = pose_df[(pose_df["video_id"] == clip["video_id"]) &
                     (pose_df["date_prefix"] == clip["date_prefix"])]
        res = process_clip(clip, pr, args.out, args.ambig_thresh, True)
        if res:
            audits.append(res)
            pose_set_by_clip[res["clip_id"]] = res["pose_set"]
            print(f"  {res['clip_id']:<34s} crops={res['n_crops']:<3d} "
                  f"assigned={res['assigned']:<3d} ambig={res['ambiguous']:<3d} "
                  f"med_score={res['median_score']} poses={res['pose_set']}")

    # update clip_index pose_set
    if "pose_set" not in cindex.columns:
        cindex["pose_set"] = ""
    cindex["pose_set"] = cindex.apply(
        lambda r: pose_set_by_clip.get(r["clip_id"], r.get("pose_set", "") or ""), axis=1)
    io_paths.write_table(cindex, os.path.join(args.out, "catalog", "clip_index"))

    # per-video audit
    if audits:
        adf = pd.DataFrame(audits)
        print("\nper-video pose assignment audit:")
        for vid, g in adf.groupby("video_id"):
            print(f"  {vid}: clips={len(g)} crops={int(g.n_crops.sum())} "
                  f"assigned={int(g.assigned.sum())} ambiguous={int(g.ambiguous.sum())} "
                  f"median_score={round(float(g.median_score.median()),3)}")
        audit_path = os.path.join(args.out, "catalog", "pose_audit")
        io_paths.write_table(adf, audit_path)
        print("wrote audit:", audit_path + ".csv")


if __name__ == "__main__":
    main()
