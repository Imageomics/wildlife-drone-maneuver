"""Shared constants and schema for the KABR test-clip library.

One place for: clip geometry, the label/index column sets, bbox-size and
vigilance thresholds, the four maneuver-suitability tags, the pose taxonomy,
and the MMLA-session -> KABR-session cross-reference used for the GT pose layer.

Nothing here imports heavy deps (no torch / pandas at import time) so it can be
pulled into any stage cheaply.
"""

from __future__ import annotations

import os

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
# Mode A (run the harness on the released dataset) needs only OUTPUT_ROOT below.
# The Mode B paths (raw-video rebuild) point at data not redistributed with this
# repo; set them via environment variables for your own machine. Defaults are
# empty so an unset Mode-B path fails loudly rather than reading a stranger's
# filesystem. See the README "Reproducing the dataset (Mode B)" section.
KABR_ROOT = os.environ.get("KABR_ROOT", "")  # KABR full-release root (Mode B)
OCCURRENCE_DIR = os.path.join(KABR_ROOT, "data", "occurrences") if KABR_ROOT else ""
VIDEO_EVENTS_CSV = os.path.join(KABR_ROOT, "data", "video_events.csv") if KABR_ROOT else ""
SESSION_EVENTS_CSV = os.path.join(KABR_ROOT, "data", "session_events.csv") if KABR_ROOT else ""

# Root for raw drone videos (Mode B). Resolve per (session/date dir, video_id)
# -- never by bare video id (DJI_xxxx recurs across sessions).
SESSION_DATA_ROOT = os.environ.get("SESSION_DATA_ROOT", "")

# Occurrence date-prefixes to drop as non-canonical. `16_01_23_flight_1` and
# `16_01_23_flight_2` are two independent annotation passes of the SAME raw
# footage (both occurrence sets embed source-SRT telemetry that matches the
# `part2` videos; their frame spans and start timestamps are identical, but the
# track-id sets are disjoint). flight_2 is the canonical pass (user-confirmed,
# 2026-06-24): it identifies more distinct individuals (e.g. DJI_0003 19 vs 11
# tracks) and flight_1's boxes do not match. Drop flight_1.
EXCLUDED_DATE_PREFIXES = frozenset({"16_01_23_flight_1"})

# Ground-truth pose crops (imageomics/KABR-poses); parent folder == label (Mode B).
POSE_LABELS_DIR = os.environ.get("POSE_LABELS_DIR", "")

# Output artifact root: where the harness reads the released clips and writes
# (re)generated labels. Override with DRONE_CLIPS_ROOT (e.g. point it at a
# `snapshot_download(...)` path of imageomics/drone-maneuver-clips).
OUTPUT_ROOT = os.environ.get("DRONE_CLIPS_ROOT", "data/drone-maneuver-clips")

# Imageomics HF dataset-card template (Mode B; build_dataset_card).
DATASET_CARD_TEMPLATE = os.environ.get("DATASET_CARD_TEMPLATE", "")

# --------------------------------------------------------------------------- #
# Clip geometry
# --------------------------------------------------------------------------- #
FPS = 30  # KABR occurrences are one row per native video frame @ 30 fps
CLIP_SECONDS = 6
CLIP_FRAMES = FPS * CLIP_SECONDS  # 180

# --------------------------------------------------------------------------- #
# Behaviour vocabulary
# --------------------------------------------------------------------------- #
# Behaviours that contribute to a vigilance signal (project_plan: run/trot/head-up).
VIGILANCE_BEHAVIOURS = frozenset({"Head Up", "Running", "Trotting"})

# Non-animal / non-informative behaviour markers seen in the occurrence files.
NON_BEHAVIOUR = frozenset({"Out of Frame", "Out of Focus", "Occluded", ""})

# --------------------------------------------------------------------------- #
# Canonical token collapses (source labels are inconsistent)
# --------------------------------------------------------------------------- #
# Applied at label-load so the published dataset + catalog use one token each.
SPECIES_CANONICAL = {"Grevy": "Grevys Zebra"}
BEHAVIOUR_CANONICAL = {"Walking": "Walk"}


def normalize_species(s) -> str:
    s = ("" if s is None else str(s)).strip()
    return SPECIES_CANONICAL.get(s, s)


def normalize_behaviour(b) -> str:
    b = ("" if b is None else str(b)).strip()
    return BEHAVIOUR_CANONICAL.get(b, b)


# --------------------------------------------------------------------------- #
# Habitat structural class
# --------------------------------------------------------------------------- #
# KABR session metadata records habitat as free text: Bitterlich relascope
# scores (a basal-area count -- higher = denser woody canopy) plus field
# remarks ("Bushy area", "Open habitat near watering hole", ...). For the clip
# library we collapse that to a structural class and keep the original verbatim
# in `habitat_notes`.
#
# Binning (user-approved 2026-06-26):
#   open   : Bitterlich 1-2, explicit "Open"
#   mixed  : Bitterlich 3-4, "Mix", scattered/"some" bushes
#   closed : Bitterlich >=5 (e.g. 10), "Bushy area/habitat"
#   unknown: bare "Near watering hole", blank
HABITAT_CLASSES = ("open", "closed", "mixed", "unknown")

# Exact source-string -> class, for the values seen in v1.0-acsos26. Keyed on
# the lower-cased, stripped string so minor casing differences still hit.
_HABITAT_CLASS_EXACT = {
    "bitterlich score: open": "open",
    "bitterlich score: 1": "open",
    "bitterlich score: 2": "open",
    "open habitat near watering hole": "open",
    "open habitat near watering hole. bitterlicht vegetation: 3": "open",
    "bitterlich score: 3": "mixed",
    "bitterlich score: 4": "mixed",
    "bitterlich score: mix (roadway)": "mixed",
    "open grassy habitat with some scattered bushes": "mixed",
    "open habitat with some bushes": "mixed",
    "bitterlich score: 10": "closed",
    "bushy area": "closed",
    "bushy habitat": "closed",
    "near watering hole": "unknown",
}


def _habitat_class_fallback(key: str) -> str:
    """Heuristic for habitat strings not in the exact map (future sites)."""
    import re

    if "mix" in key:
        return "mixed"
    m = re.search(r"bitterlich\w*\D*(\d+)", key)
    if m:
        n = int(m.group(1))
        if n <= 2:
            return "open"
        if n <= 4:
            return "mixed"
        return "closed"
    if "bushy" in key:
        return "closed"
    if "bush" in key:  # "some bushes", "scattered bushes"
        return "mixed"
    if "open" in key:
        return "open"
    return "unknown"


def normalize_habitat(raw) -> tuple[str, str]:
    """Map free-text habitat metadata to ``(structural_class, notes)``.

    ``structural_class`` is one of :data:`HABITAT_CLASSES`; ``notes`` preserves
    the original verbatim string (empty when the source was blank).
    """
    s = ("" if raw is None else str(raw)).strip()
    if not s or s.lower() == "nan":
        return "unknown", ""
    key = s.lower()
    return _HABITAT_CLASS_EXACT.get(key, _habitat_class_fallback(key)), s

# --------------------------------------------------------------------------- #
# Bounding-box size classes (fraction of frame area)
# --------------------------------------------------------------------------- #
# far  : bbox_area_frac <  BBOX_FAR_MAX
# close: bbox_area_frac >= BBOX_CLOSE_MIN
# medium: in between
#
# Thresholds are RELATIVE to this dataset's range. KABR is flown at 20-50 m, so
# animals occupy a small frame fraction throughout (p99 of bbox area ~= 0.018).
# These cut points (~p65 / ~p92 of the bbox-area distribution) give a usable
# far/medium/close split; "close" here means close *for this survey*, not that
# the animal fills the frame.
BBOX_FAR_MAX = 0.001
BBOX_CLOSE_MIN = 0.005


def bbox_size_class(area_frac: float) -> str:
    if area_frac < BBOX_FAR_MAX:
        return "far"
    if area_frac >= BBOX_CLOSE_MIN:
        return "close"
    return "medium"


# --------------------------------------------------------------------------- #
# Maneuver-suitability tags
# --------------------------------------------------------------------------- #
MANEUVERS = ("launch", "follow", "behavior_adaptive", "soi_aware")

# --------------------------------------------------------------------------- #
# Pose taxonomy (matches imageomics/KABR-poses folder names)
# --------------------------------------------------------------------------- #
POSE_CLASSES = (
    "front",
    "front-left",
    "front-right",
    "left",
    "right",
    "back-left",
    "back-right",
    "back",
)

# --------------------------------------------------------------------------- #
# MMLA session  ->  KABR session cross-reference (user-provided).
# Used only for the GT pose layer: a pose crop names its MMLA session + video,
# which we map to the KABR session (and thus the right date-prefixed occurrence
# file / raw video), since DJI_0002 and DJI_0006 recur across sessions.
# --------------------------------------------------------------------------- #
MMLA_TO_KABR = {
    "mpala_session_1": {
        "kabr_session": "KABR-2023:12_01_23_session_4",
        "date_prefix": "12_01_23",
        "videos": ("DJI_0001", "DJI_0002"),
        "species": "Giraffe",
    },
    "mpala_session_2": {
        "kabr_session": "KABR-2023:17_01_2023_session_1",
        "date_prefix": "17_01_2023_session_1",
        "videos": ("DJI_0005", "DJI_0006"),
        "species": "Plains zebra",
    },
    "mpala_session_3": {
        "kabr_session": "KABR-2023:18_01_2023_session_7",
        "date_prefix": "18_01_2023_session_7",
        "videos": ("DJI_0068", "DJI_0069", "DJI_0070", "DJI_0071"),
        "species": "Grevy's zebra",
    },
    "mpala_session_4": {
        "kabr_session": "KABR-2023:20_01_2023_session_3",
        "date_prefix": "20_01_2023_session_3",
        "videos": ("DJI_0142", "DJI_0143", "DJI_0144", "DJI_0145", "DJI_0146", "DJI_0147"),
        "species": "Grevy's zebra",
    },
    "mpala_session_5": {
        "kabr_session": "KABR-2023:21_01_2023_session_5",
        "date_prefix": "21_01_2023_session_5",
        "videos": ("DJI_0206", "DJI_0208", "DJI_0210", "DJI_0211"),
        "species": "Giraffe, Plains and Grevy's zebras",
    },
}

# Videos that actually carry GT pose crops (validated: 691 crops).
POSE_VIDEOS = ("DJI_0002", "DJI_0006", "DJI_0070", "DJI_0145", "DJI_0208")

# --------------------------------------------------------------------------- #
# Column sets
# --------------------------------------------------------------------------- #
# One row per frame per track.
LABEL_COLUMNS = [
    "clip_id",
    "video_id",
    "session_id",
    "frame_global",
    "frame_local",
    "time_s",
    "track_id",
    "species",
    "behaviour",
    "vigilant",
    "pose",
    "pose_provenance",
    "pose_match_score",
    "individual_id",
    "xtl",
    "ytl",
    "xbr",
    "ybr",
    "x_c",
    "y_c",
    "w",
    "h",
    "bbox_area_frac",
    "bbox_size_class",
    "occluded",
    "outside",
    "latitude",
    "longitude",
    "altitude",
    "date_time",
]

# One row per clip.
CLIP_INDEX_COLUMNS = [
    "clip_id",
    "video_id",
    "session_id",
    "start_frame",
    "end_frame",
    "start_time",
    "species_set",
    "habitat",
    "habitat_notes",
    "herd_size",
    "n_tracks",
    "behaviours_present",
    "has_vigilance",
    "bbox_size_classes",
    "pose_set",
    "suitable_maneuvers",
    "fair2_video_eventID",
    "fair2_session_eventID",
    "source_video_path",
    "label_provenance",
]
