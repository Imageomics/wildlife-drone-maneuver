"""Stage 2: select 6-second windows that exercise the maneuvers.

Slides non-overlapping (configurable) windows over the frames that carry tracks,
scores each window against the four maneuver tags, then runs a coverage pass so
the kept set spans species x bbox-size x maneuver x habitat. Pose-labeled
windows (5 videos) are force-included so the GT pose layer has clips.

Outputs `catalog/clip_index.csv` + `catalog/coverage_report.md`.

Usage:
    python -m clip_library.select_clips [--out DIR] [--stride N] [--per-video-cap K]
"""

from __future__ import annotations

import os
import argparse
import collections

import numpy as np
import pandas as pd

from . import schema, io_paths, pose_gt

MIN_PRESENT_FRAC = 0.5     # >= this fraction of window frames must contain a track
FOLLOW_MULTI_FRAC = 0.8    # >= this fraction with >=2 tracks => follow
LAUNCH_GROWTH = 1.5        # last-third / first-third max-area ratio => approach
SOI_MIN_AREA = schema.BBOX_FAR_MAX  # animals big enough to read a pose


def _track_rows(occ: pd.DataFrame, w: int, h: int) -> pd.DataFrame:
    """Track rows with bbox geometry; sorted by frame."""
    t = occ[occ["xtl"].notna()].copy()
    for c in ("xtl", "ytl", "xbr", "ybr", "frame"):
        t[c] = pd.to_numeric(t[c], errors="coerce")
    t = t.dropna(subset=["xtl", "ytl", "xbr", "ybr", "frame"])
    t["frame"] = t["frame"].astype(int)
    # collapse ~10x exact-duplicate rows per (frame, track) from the merge.
    t = t.drop_duplicates(subset=["frame", "id", "xtl", "ytl", "xbr", "ybr", "behaviour"])
    area = (t["xbr"] - t["xtl"]).clip(lower=0) * (t["ybr"] - t["ytl"]).clip(lower=0)
    t["area_frac"] = area / float(w * h) if w and h else 0.0
    t["size_class"] = t["area_frac"].map(schema.bbox_size_class)
    return t.sort_values("frame").reset_index(drop=True)


def _score_window(sub: pd.DataFrame, s: int, e: int) -> dict:
    """Score one window [s, e); return features + maneuver tags."""
    n = e - s
    # per-frame aggregates
    g = sub.groupby("frame")
    ntracks = g["id"].nunique()
    maxarea = g["area_frac"].max()
    nvig = g["behaviour"].apply(lambda b: b.isin(schema.VIGILANCE_BEHAVIOURS).sum())
    frames_present = set(ntracks.index)

    present_frac = len(frames_present) / n
    # reindex to per-third arrays for trend tests
    thirds = np.array_split(np.arange(s, e), 3)

    def third_mean(series, idxs):
        vals = [series.get(i, 0) for i in idxs]
        return float(np.mean(vals)) if len(vals) else 0.0

    a_first = third_mean(maxarea, thirds[0])
    a_last = third_mean(maxarea, thirds[2])
    v_first = third_mean(nvig, thirds[0])
    v_last = third_mean(nvig, thirds[2])

    multi_frac = (ntracks >= 2).sum() / n
    tags = []

    # launch: target approaches (max bbox area grows, reaching visible size)
    if a_last >= SOI_MIN_AREA and a_first > 0 and a_last / max(a_first, 1e-9) >= LAUNCH_GROWTH:
        tags.append("launch")
    elif a_first < SOI_MIN_AREA and a_last >= schema.BBOX_CLOSE_MIN:
        tags.append("launch")

    # follow: herd present throughout
    if multi_frac >= FOLLOW_MULTI_FRAC:
        tags.append("follow")

    # behavior_adaptive: vigilance present (bonus weight if calm->vigilant)
    if int(nvig.sum()) > 0:
        tags.append("behavior_adaptive")

    # soi_aware proxy: >=2 tracks, big enough to read pose (pose stage refines)
    if multi_frac >= MIN_PRESENT_FRAC and float(maxarea.median()) >= SOI_MIN_AREA:
        tags.append("soi_aware")

    return {
        "present_frac": round(present_frac, 3),
        "n_tracks_max": int(ntracks.max()) if len(ntracks) else 0,
        "multi_frac": round(float(multi_frac), 3),
        "max_area_frac": round(float(maxarea.max()) if len(maxarea) else 0.0, 4),
        "vigilant_to_calm_transition": bool(v_first == 0 and v_last > 0),
        "species_set": "|".join(
            sorted({schema.normalize_species(s) for s in sub["label"].dropna()})
        ),
        "behaviours_present": "|".join(
            sorted({schema.normalize_behaviour(b) for b in sub["behaviour"].dropna()}
                   - schema.NON_BEHAVIOUR)
        ),
        "bbox_size_classes": "|".join(sorted(sub["size_class"].unique())),
        "suitable_maneuvers": "|".join(tags),
        "_tags": tags,
    }


def candidate_windows(vrow: pd.Series, stride: int, pose_frames: set) -> list[dict]:
    """All scored, tag-bearing windows for one video."""
    occ = pd.read_csv(vrow["occurrence_path"], low_memory=False)
    w, h = int(vrow["width"]), int(vrow["height"])
    if not (w and h):
        return []
    t = _track_rows(occ, w, h)
    if t.empty:
        return []
    frames_arr = t["frame"].values
    fmin, fmax = int(vrow["frame_min_with_tracks"]), int(vrow["frame_max_with_tracks"])

    out = []
    for s in range(fmin, fmax - schema.CLIP_FRAMES + 2, stride):
        e = s + schema.CLIP_FRAMES
        lo = int(np.searchsorted(frames_arr, s, "left"))
        hi = int(np.searchsorted(frames_arr, e, "left"))
        if hi - lo == 0:
            continue
        sub = t.iloc[lo:hi]
        feat = _score_window(sub, s, e)
        if feat["present_frac"] < MIN_PRESENT_FRAC or not feat["_tags"]:
            continue
        pose_covered = len(pose_frames & set(range(s, e))) if pose_frames else 0
        _habitat_class, _habitat_notes = schema.normalize_habitat(vrow.get("habitat", ""))
        clip_id = f"{vrow['date_prefix']}-{vrow['video_id']}_{s:06d}"
        rec = {
            "clip_id": clip_id,
            "video_id": vrow["video_id"],
            "session_id": vrow["session_id"],
            "date_prefix": vrow["date_prefix"],
            "start_frame": s,
            "end_frame": e,
            "start_time": round(s / schema.FPS, 2),
            # Structural class (open/closed/mixed/unknown) + verbatim original.
            "habitat": _habitat_class,
            "habitat_notes": _habitat_notes,
            "herd_size": vrow.get("herd_size", ""),
            "pose_frames_covered": pose_covered,
            "fair2_video_eventID": vrow.get("fair2_video_eventID", ""),
            "fair2_session_eventID": vrow.get("fair2_session_eventID", ""),
            "source_video_path": vrow.get("source_video_path", ""),
        }
        rec.update({k: v for k, v in feat.items() if not k.startswith("_")})
        rec["_tags"] = feat["_tags"]
        out.append(rec)
    return out


def select(candidates: list[dict], per_video_cap: int, pose_cap: int) -> list[dict]:
    """Force-include pose windows (capped), then greedily fill coverage cells."""
    chosen: dict[str, dict] = {}
    per_video = collections.Counter()

    def take(rec):
        chosen[rec["clip_id"]] = rec
        per_video[rec["video_id"]] += 1

    # 1) force-include the best pose-covering windows per pose video, capped so
    #    the 5 pose videos don't crowd out maneuver/species/session coverage.
    pose_cands = [c for c in candidates if c["pose_frames_covered"] > 0]
    pose_cands.sort(key=lambda c: c["pose_frames_covered"], reverse=True)
    for c in pose_cands:
        if per_video[c["video_id"]] >= pose_cap:
            continue
        take(c)

    # 2) greedy coverage over (species x size-class x maneuver), habitats, sessions
    covered_cells, covered_habitats, covered_sessions = set(), set(), set()
    for c in chosen.values():
        _register_cells(c, covered_cells, covered_habitats, covered_sessions)

    rest = [c for c in candidates if c["clip_id"] not in chosen]
    rest.sort(key=lambda c: (len(c["_tags"]), c["n_tracks_max"], c["present_frac"]), reverse=True)
    for c in rest:
        if per_video[c["video_id"]] >= per_video_cap:
            continue
        cells, habs, sess = set(), set(), set()
        _register_cells(c, cells, habs, sess)
        if (cells - covered_cells) or (habs - covered_habitats) or (sess - covered_sessions):
            take(c)
            covered_cells |= cells
            covered_habitats |= habs
            covered_sessions |= sess

    return list(chosen.values())


def _register_cells(c: dict, cells: set, habitats: set, sessions: set):
    species = [s for s in c["species_set"].split("|") if s]
    sizes = [s for s in c["bbox_size_classes"].split("|") if s]
    for sp in species or ["?"]:
        for sz in sizes or ["?"]:
            for mv in c["_tags"]:
                cells.add((sp, sz, mv))
    if str(c.get("habitat", "")):
        habitats.add(str(c["habitat"]))
    if str(c.get("session_id", "")):
        sessions.add(str(c["session_id"]))


def coverage_report(clips: list[dict], stride: int, per_video_cap: int) -> str:
    sp = collections.Counter()
    sz = collections.Counter()
    mv = collections.Counter()
    hab = collections.Counter()
    pose_clips = sum(1 for c in clips if c["pose_frames_covered"] > 0)
    for c in clips:
        for s in c["species_set"].split("|"):
            if s:
                sp[s] += 1
        for s in c["bbox_size_classes"].split("|"):
            if s:
                sz[s] += 1
        for m in c["_tags"]:
            mv[m] += 1
        if str(c.get("habitat", "")):
            hab[str(c["habitat"])] += 1
    lines = ["# Clip coverage report", ""]
    lines.append(f"- total clips: **{len(clips)}** (stride={stride}, per-video cap={per_video_cap})")
    lines.append(f"- clips covering GT-pose frames: **{pose_clips}** (5 pose videos)")
    lines.append(f"- distinct videos: {len(set(c['video_id'] for c in clips))}")
    for title, ctr in [("Species", sp), ("Bbox size class", sz),
                       ("Maneuver tag", mv), ("Habitat", hab)]:
        lines += ["", f"## {title}"]
        for k, v in ctr.most_common():
            lines.append(f"- {k}: {v}")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description="Stage 2: select maneuver clips")
    ap.add_argument("--out", default=schema.OUTPUT_ROOT)
    ap.add_argument("--stride", type=int, default=schema.CLIP_FRAMES,
                    help="window stride in frames (default = clip length, non-overlapping)")
    ap.add_argument("--per-video-cap", type=int, default=8)
    ap.add_argument("--pose-cap", type=int, default=5,
                    help="max force-included pose windows per pose video")
    args = ap.parse_args()

    vindex = io_paths.read_table(os.path.join(args.out, "catalog", "video_index"))
    usable = vindex[(vindex["video_status"] == "ok") & (vindex["n_frames_with_tracks"] > 0)]
    print(f"Selecting from {len(usable)} usable videos (of {len(vindex)}) ...")

    pf = pose_gt.pose_frames_by_video()
    candidates = []
    for _, vrow in usable.iterrows():
        cw = candidate_windows(vrow, args.stride, pf.get(vrow["video_id"], set()))
        candidates.extend(cw)
        if cw:
            print(f"  {vrow['video_id']:>9s} ({vrow['date_prefix']:<22s}) -> {len(cw)} candidate windows")
    print(f"\n{len(candidates)} candidate windows; selecting for coverage ...")

    clips = select(candidates, args.per_video_cap, args.pose_cap)
    clips.sort(key=lambda c: c["clip_id"])

    # write clip_index (drop private _tags)
    df = pd.DataFrame([{k: v for k, v in c.items() if not k.startswith("_")} for c in clips])
    # column ordering: known first, extras after
    out_index = os.path.join(args.out, "catalog", "clip_index")
    paths = io_paths.write_table(df, out_index)

    report = coverage_report(clips, args.stride, args.per_video_cap)
    report_path = os.path.join(args.out, "catalog", "coverage_report.md")
    with open(report_path, "w") as fh:
        fh.write(report)

    print(f"\nselected {len(clips)} clips")
    print(report)
    print("wrote:", ", ".join(paths), "and", report_path)


if __name__ == "__main__":
    main()
