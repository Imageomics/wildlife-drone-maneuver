"""Stage 1: scan the KABR occurrence files into a per-video catalog.

Produces `catalog/video_index.csv` -- one row per occurrence file with a
frame-level summary (which frames carry tracks, behaviours present, vigilance,
max tracks/frame), the resolved raw-video path + status, video pixel
dimensions, and the joined Darwin-Core / FAIR^2 metadata (habitat, species,
herd size, event IDs). This is the input to stage 2 (clip selection).

Usage:
    python -m clip_library.scan_sessions [--out DIR] [--limit N]
"""

from __future__ import annotations

import os
import argparse

import pandas as pd

from . import schema, io_paths


def _first_date(occ: pd.DataFrame) -> str:
    """Occurrence `date` is DD_MM_YY; return it (or '')."""
    if "date" in occ.columns and len(occ):
        return str(occ["date"].iloc[0])
    return ""


def _ddmmyy_to_iso(dmy: str) -> str:
    """'18_01_23' / '18_01_2023' -> '2023-01-18' (best effort)."""
    parts = dmy.split("_")
    if len(parts) != 3:
        return ""
    d, m, y = parts
    if len(y) == 2:
        y = "20" + y
    return f"{y}-{int(m):02d}-{int(d):02d}"


def _match_video_event(ve: pd.DataFrame, video_id: str, iso_date: str):
    """Find the video_events row for this (video_id, date)."""
    cand = ve[ve["eventID"].str.endswith(":" + video_id, na=False)]
    if len(cand) == 0:
        return None
    if len(cand) > 1 and iso_date:
        by_date = cand[cand["eventDate"].astype(str) == iso_date]
        if len(by_date):
            cand = by_date
    return cand.iloc[0]


def build_video_index(limit: int | None = None) -> pd.DataFrame:
    ve = pd.read_csv(schema.VIDEO_EVENTS_CSV)
    se = pd.read_csv(schema.SESSION_EVENTS_CSV)
    se_by_id = se.set_index("eventID")

    rows = []
    files = io_paths.occurrence_files()
    if limit:
        files = files[:limit]

    for f in files:
        date_prefix, video_id = io_paths.parse_occurrence_name(f)
        occ = pd.read_csv(f, low_memory=False)
        tracks = occ[occ["xtl"].notna()].copy()

        # frame-level summary
        if len(tracks):
            frames = tracks["frame"].astype(int)
            tracks_per_frame = tracks.groupby("frame")["id"].nunique()
            beh = sorted(
                {schema.normalize_behaviour(b) for b in tracks["behaviour"].dropna()}
                - schema.NON_BEHAVIOUR
            )
            species = sorted({schema.normalize_species(s) for s in tracks["label"].dropna()})
            has_vig = bool(set(beh) & schema.VIGILANCE_BEHAVIOURS)
            frame_min, frame_max = int(frames.min()), int(frames.max())
            n_frames_tracks = int(frames.nunique())
            max_tpf = int(tracks_per_frame.max())
        else:
            beh, species, has_vig = [], [], False
            frame_min = frame_max = n_frames_tracks = max_tpf = 0

        # raw video resolution + dims (occ telemetry disambiguates part1/part2)
        res = io_paths.resolve_raw_video(date_prefix, video_id, occ_csv=f)
        if res.ok:
            try:
                w, h, fps, nframes = io_paths.video_dims(res.path)
            except Exception:
                w = h = 0
                fps = schema.FPS
                nframes = 0
        else:
            w = h = nframes = 0
            fps = schema.FPS

        # metadata join
        iso = _ddmmyy_to_iso(_first_date(occ))
        vev = _match_video_event(ve, video_id, iso)
        if vev is not None:
            video_eventID = vev["eventID"]
            session_eventID = vev["parentEventID"]
            min_elev = vev.get("minimumElevationInMeters")
            max_elev = vev.get("maximumElevationInMeters")
        else:
            video_eventID = ""
            session_eventID = ""
            min_elev = max_elev = None

        habitat = species_common = herd_size = locality = ""
        if session_eventID and session_eventID in se_by_id.index:
            srow = se_by_id.loc[session_eventID]
            habitat = str(srow.get("habitat", "") or "")
            species_common = str(srow.get("_species_common", "") or "")
            herd_size = srow.get("organismQuantity", "")
            locality = str(srow.get("locality", "") or "")

        rows.append(
            {
                "video_id": video_id,
                "date_prefix": date_prefix,
                "session_id": session_eventID,
                "occurrence_path": f,
                "source_video_path": res.path or "",
                "video_status": res.status,
                "video_note": res.note,
                "video_candidates": "|".join(res.candidates),
                "width": w,
                "height": h,
                "fps": round(float(fps), 3),
                "n_frames_total": nframes,
                "n_frames_with_tracks": n_frames_tracks,
                "frame_min_with_tracks": frame_min,
                "frame_max_with_tracks": frame_max,
                "max_tracks_per_frame": max_tpf,
                "species": "|".join(species),
                "behaviours_present": "|".join(beh),
                "has_vigilance": has_vig,
                "fair2_video_eventID": video_eventID,
                "fair2_session_eventID": session_eventID,
                "habitat": habitat,
                "species_common": species_common,
                "herd_size": herd_size,
                "locality": locality,
                "min_elev_m": min_elev,
                "max_elev_m": max_elev,
            }
        )
        print(
            f"  {video_id:>9s} {date_prefix:<28s} tracks_frames={n_frames_tracks:>6d} "
            f"max_tpf={max_tpf:>2d} vig={int(has_vig)} video={res.status}"
        )

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="Stage 1: scan KABR sessions -> video_index")
    ap.add_argument("--out", default=schema.OUTPUT_ROOT, help="output artifact root")
    ap.add_argument("--limit", type=int, default=None, help="only first N occurrence files")
    args = ap.parse_args()

    print("Scanning KABR occurrence files ...")
    df = build_video_index(limit=args.limit)
    out = os.path.join(args.out, "catalog", "video_index")
    paths = io_paths.write_table(df, out)

    # summary
    n = len(df)
    ok = int((df["video_status"] == "ok").sum())
    vig = int(df["has_vigilance"].sum())
    print(
        f"\nvideo_index: {n} videos | raw resolved {ok}/{n} "
        f"({(df['video_status'] != 'ok').sum()} need attention) | "
        f"{vig} with vigilance behaviours"
    )
    unresolved = df[df["video_status"] != "ok"][["video_id", "date_prefix", "video_status", "video_note"]]
    if len(unresolved):
        print("  videos needing user attention:")
        for _, r in unresolved.iterrows():
            print(f"    {r.video_id} ({r.date_prefix}): {r.video_status} -- {r.video_note}")
    print("wrote:", ", ".join(paths))


if __name__ == "__main__":
    main()
