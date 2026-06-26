"""Load the KABR-poses ground-truth crops into a tidy table.

Each crop's filename encodes (mmla_session, video, partition, frame, class_id,
det_index); the parent folder is the 8-class pose label. We keep only the
`mpala_*` crops (KABR); `opc_*` (Ol Pejeta) are skipped. The MMLA session maps
to a KABR session/date via schema.MMLA_TO_KABR.
"""

from __future__ import annotations

import os
import glob

import pandas as pd

from . import schema, pose_backtrack


def load_pose_crops() -> pd.DataFrame:
    """Return a DataFrame of GT pose crops, one row per crop."""
    rows = []
    for class_dir in sorted(glob.glob(os.path.join(schema.POSE_LABELS_DIR, "*"))):
        if not os.path.isdir(class_dir):
            continue
        label = os.path.basename(class_dir)
        if label.startswith("_") or label not in schema.POSE_CLASSES:
            continue
        for crop in glob.glob(os.path.join(class_dir, "*.jpg")):
            name = os.path.basename(crop)
            if not name.startswith("mpala"):
                continue  # skip opc / non-KABR
            try:
                m = pose_backtrack.parse_crop_name(name)
            except ValueError:
                continue
            mapping = schema.MMLA_TO_KABR.get(m.session, {})
            rows.append(
                {
                    "pose_label": label,
                    "mmla_session": m.session,
                    "video_id": m.video,
                    "partition": m.partition,
                    "frame": m.frame,
                    "class_id": m.class_id,
                    "det_index": m.det_index,
                    "kabr_session": mapping.get("kabr_session", ""),
                    "date_prefix": mapping.get("date_prefix", ""),
                    "crop_path": crop,
                }
            )
    return pd.DataFrame(rows)


def pose_frames_by_video() -> dict[str, set]:
    """{video_id: {frame, ...}} of GT-pose-labeled frames."""
    df = load_pose_crops()
    out: dict[str, set] = {}
    for vid, grp in df.groupby("video_id"):
        out[vid] = set(int(f) for f in grp["frame"].unique())
    return out


if __name__ == "__main__":
    df = load_pose_crops()
    print(f"{len(df)} KABR pose crops")
    print(df.groupby(["video_id", "date_prefix"]).size())
    print("\nframes per video:", {k: len(v) for k, v in pose_frames_by_video().items()})
