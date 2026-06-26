"""Path resolution and small I/O helpers for the clip library.

The important piece here is `resolve_raw_video`: the same `DJI_xxxx` filename
recurs across sessions and OSC's per-session folders are inconsistent, so we
resolve a raw video by (session/date prefix, video_id) under SESSION_DATA_ROOT
and refuse to guess when the match is ambiguous.
"""

from __future__ import annotations

import os
import glob
from dataclasses import dataclass, field
from typing import Optional

from . import schema


# --------------------------------------------------------------------------- #
# Occurrence files
# --------------------------------------------------------------------------- #
def occurrence_files() -> list[str]:
    """All canonical occurrence CSV paths, sorted.

    Drops files whose date-prefix is in `schema.EXCLUDED_DATE_PREFIXES`
    (non-canonical annotation passes; see schema for the rationale).
    """
    files = sorted(glob.glob(os.path.join(schema.OCCURRENCE_DIR, "*.csv")))
    excluded = schema.EXCLUDED_DATE_PREFIXES
    if not excluded:
        return files
    return [f for f in files if parse_occurrence_name(f)[0] not in excluded]


def parse_occurrence_name(path: str) -> tuple[str, str]:
    """`{date_prefix}-{video_id}.csv` -> (date_prefix, video_id).

    The date_prefix may itself contain hyphens are absent here, but video ids
    look like `DJI_0070`; we split on the LAST '-' before the video token.
    """
    base = os.path.basename(path)[:-4]  # strip .csv
    # video id is the trailing 'DJI_####'
    idx = base.rfind("-DJI_")
    if idx == -1:
        # fall back: last hyphen
        idx = base.rfind("-")
    date_prefix = base[:idx]
    video_id = base[idx + 1 :]
    return date_prefix, video_id


# --------------------------------------------------------------------------- #
# Raw video resolution
# --------------------------------------------------------------------------- #
@dataclass
class VideoResolution:
    video_id: str
    date_prefix: str
    path: Optional[str] = None          # chosen path, or None
    candidates: list[str] = field(default_factory=list)
    status: str = "unresolved"           # 'ok' | 'ambiguous' | 'missing'
    note: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def _session_dir_candidates(date_prefix: str) -> list[str]:
    """session_data subdirs matching a date/session prefix."""
    root = schema.SESSION_DATA_ROOT
    if not os.path.isdir(root):
        return []
    exact = os.path.join(root, date_prefix)
    if os.path.isdir(exact):
        return [exact]
    # fall back: dirs sharing the leading date token (e.g. '16_01_23_flight_1' -> '16_01_23')
    lead = date_prefix.split("_session")[0].split("_flight")[0]
    out = []
    for name in sorted(os.listdir(root)):
        full = os.path.join(root, name)
        if os.path.isdir(full) and (name == lead or name.startswith(lead)):
            out.append(full)
    return out


def _srt_start_timestamp(mp4_path: str) -> Optional[str]:
    """First wall-clock timestamp ('YYYY-MM-DD HH:MM:SS') from the sibling .SRT.

    DJI SRT files carry per-frame telemetry; the first frame's timestamp is a
    unique fingerprint of which raw sortie this file is.
    """
    import re

    srt = os.path.splitext(mp4_path)[0] + ".SRT"
    if not os.path.exists(srt):
        srt = os.path.splitext(mp4_path)[0] + ".srt"
    if not os.path.exists(srt):
        return None
    with open(srt, errors="ignore") as fh:
        m = re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", fh.read(4096))
    return m.group(1) if m else None


def occurrence_start_timestamp(occ_csv: str) -> Optional[str]:
    """First non-empty `date_time` ('YYYY-MM-DD HH:MM:SS') in an occurrence CSV.

    Occurrence files copy the source video's SRT telemetry per frame, so this
    matches exactly one raw candidate's `_srt_start_timestamp`.
    """
    import csv

    try:
        with open(occ_csv, errors="ignore") as fh:
            for row in csv.DictReader(fh):
                dt = (row.get("date_time") or "").split(",")[0].strip()
                if dt:
                    return dt
    except (OSError, csv.Error):
        return None
    return None


def resolve_raw_video(
    date_prefix: str, video_id: str, occ_csv: Optional[str] = None
) -> VideoResolution:
    """Locate the raw .MP4 for (date_prefix, video_id) under SESSION_DATA_ROOT.

    Never guesses blindly. When several raw files share the `DJI_xxxx` name
    (e.g. part1/part2 sorties) and `occ_csv` is given, disambiguate by matching
    the occurrence's embedded SRT start-timestamp to each candidate's .SRT --
    an exact, deterministic match. Otherwise returns 'ambiguous'/'missing' for
    the caller to surface.
    """
    res = VideoResolution(video_id=video_id, date_prefix=date_prefix)
    dirs = _session_dir_candidates(date_prefix)
    if not dirs:
        res.status = "missing"
        res.note = f"no session_data dir for prefix '{date_prefix}'"
        return res

    hits: list[str] = []
    for d in dirs:
        for ext in ("MP4", "mp4", "MOV", "mov"):
            hits.extend(glob.glob(os.path.join(d, "**", f"{video_id}.{ext}"), recursive=True))
    # de-dup, keep deterministic order
    hits = sorted(set(hits))
    res.candidates = hits

    if not hits:
        res.status = "missing"
        res.note = f"no {video_id}.MP4 under {[os.path.basename(d) for d in dirs]}"
    elif len(hits) == 1:
        res.status = "ok"
        res.path = hits[0]
    elif occ_csv:
        occ_ts = occurrence_start_timestamp(occ_csv)
        matches = [h for h in hits if occ_ts and _srt_start_timestamp(h) == occ_ts]
        if len(matches) == 1:
            res.status = "ok"
            res.path = matches[0]
            res.note = f"telemetry-matched SRT start {occ_ts} ({len(hits)} candidates)"
        else:
            res.status = "ambiguous"
            res.note = (
                f"{len(hits)} candidates; SRT-telemetry match resolved "
                f"{len(matches)} (occ_ts={occ_ts})"
            )
    else:
        res.status = "ambiguous"
        res.note = f"{len(hits)} candidate files; needs user disambiguation"
    return res


# --------------------------------------------------------------------------- #
# Video frame I/O (cv2)
# --------------------------------------------------------------------------- #
def video_dims(path: str) -> tuple[int, int, float, int]:
    """(width, height, fps, n_frames) for a video via cv2."""
    import cv2

    cap = cv2.VideoCapture(path)
    try:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or schema.FPS
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    return w, h, fps, n


def read_frame(path: str, frame_idx: int):
    """Read a single BGR frame by absolute index (None if unavailable)."""
    import cv2

    cap = cv2.VideoCapture(path)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
    finally:
        cap.release()
    return frame if ok else None


# --------------------------------------------------------------------------- #
# Table I/O (CSV always; Parquet when pyarrow is present)
# --------------------------------------------------------------------------- #
def have_parquet() -> bool:
    try:
        import pyarrow  # noqa: F401

        return True
    except Exception:
        return False


def write_table(df, path_noext: str) -> list[str]:
    """Write a DataFrame to `<path_noext>.csv` (+ `.parquet` if available)."""
    os.makedirs(os.path.dirname(path_noext), exist_ok=True)
    written = []
    csv_path = path_noext + ".csv"
    df.to_csv(csv_path, index=False)
    written.append(csv_path)
    if have_parquet():
        pq_path = path_noext + ".parquet"
        try:
            df.to_parquet(pq_path, index=False)
            written.append(pq_path)
        except Exception:
            pass
    return written


def read_table(path_noext: str):
    """Read `<path_noext>.parquet` if present else `.csv`."""
    import pandas as pd

    if have_parquet() and os.path.exists(path_noext + ".parquet"):
        return pd.read_parquet(path_noext + ".parquet")
    return pd.read_csv(path_noext + ".csv", low_memory=False)
