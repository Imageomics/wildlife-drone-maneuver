"""Deterministic maneuver-label generator (the ACSOS replay harness).

Given a clip's per-frame-per-track ``labels.csv`` and a maneuver + user params,
this replays the formal decision tree in ``maneuver_decision_tree.md`` (shipped
alongside this module) and emits one ground-truth drone action per frame. The
action is a *set* drawn
from the 9-action space, smoothed by a trailing rolling average so the GT track
isn't jerky.

This is the artifact's headline contribution: a small, inspectable policy
specification executed deterministically over a labeled benchmark, so a learned
navigation policy can be scored against an expert-calibrated reference.

Action space (per frame, a set; empty -> {hover}):
    up, down, forward, back, left, right, yaw-left, yaw-right, hover

Usage:
    # one clip, one maneuver
    python -m clip_library.maneuver_labels --clip 12_01_23-DJI_0002_000745 --maneuver track
    # all extracted clips, each maneuver it's tagged suitable for
    python -m clip_library.maneuver_labels --all
"""
from __future__ import annotations

import os
import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import schema, io_paths

# --------------------------------------------------------------------------- #
# Frame geometry & the keep-in zone (center 50%)
# --------------------------------------------------------------------------- #
FRAME_W, FRAME_H = 3840, 2160
KEEP_LO, KEEP_HI = 0.25, 0.75          # trim 1/4 off each edge

# Smoothing: decompose actions onto signed axes, trailing-average over W frames,
# then re-threshold. |mean| <= AXIS_DEADZONE -> no motion on that axis.
SMOOTH_WINDOW = 90                      # frames (3 s @ 30 fps)
AXIS_DEADZONE = 0.33

# Behaviour -> vigilance (matches schema.VIGILANCE_BEHAVIOURS)
VIGILANCE_WINDOW = 90                  # frames (3 s) for the smoothed S_t
HOVER_HOLD = 150                       # frames (5 s) BAF hovers after a trigger

# Pose ring, ordered so a +1 step == one 'yaw-left' increment of apparent pose
# (front -> front-right -> right ...). To rotate toward a target SoI we take the
# shortest signed distance around this ring; sign convention is OPEN ITEM 7.
POSE_RING = ["front", "front-right", "right", "back-right",
             "back", "back-left", "left", "front-left"]


# --------------------------------------------------------------------------- #
# User parameters (reviewer-tunable; defaults from the spec)
# --------------------------------------------------------------------------- #
@dataclass
class Params:
    # APPROACH
    launch_altitude: float = 50.0
    end_altitude: float = 30.0
    target_species: str | None = None   # None = any species counts as "target"
    # TRACK / SoI pixel targets (longest bbox side, px)
    desired_pixels: float = 30.0        # TRACK default; SoI overrides per objective
    pixel_band: float = 0.25            # +/- tolerance around desired_pixels
    max_animals: int = 5                # follow the N largest bboxes
    # BAF
    theta_S: float = 0.5                # vigilance threshold on smoothed S_t
    baf_response: str = "retreat"       # 'retreat' -> {back, up} | 'hover'
    # SoI
    soi: str = "left"                   # desired pose (broadside by default)

    def with_objective(self, objective: str) -> "Params":
        """SoI desired-pixel presets by downstream objective (cite prior work)."""
        px = {"track": 30.0, "behavior": 100.0, "reid": 500.0}.get(objective)
        if px is not None:
            self.desired_pixels = px
        return self


# Reviewer-tunable thresholds the UI / `replay()` expose, with their default
# values. Keys are the names the harness API uses in its `thresholds` dict;
# values are sourced from `Params` + the smoothing constant (single source of
# truth). Any subset may be passed to `replay()`; missing keys fall back here.
DEFAULT_THRESHOLDS = {
    "theta_S": Params().theta_S,            # vigilance threshold on smoothed S_t
    "desired_bbox_px": Params().desired_pixels,  # target longest-bbox side (px)
    "soi_pose": Params().soi,               # desired Surface-of-Interest pose
    "smoothing_window": SMOOTH_WINDOW,      # trailing rolling-average window (frames)
}


# axis encoding: (dx, dy, dz, dyaw); +x right, +y up, +z forward, +yaw yaw-right
_AXIS_TO_TOKENS = {
    ("x", +1): "right", ("x", -1): "left",
    ("y", +1): "up",    ("y", -1): "down",
    ("z", +1): "forward", ("z", -1): "back",
    ("yaw", +1): "yaw-right", ("yaw", -1): "yaw-left",
}
_TOKEN_TO_AXIS = {tok: (ax, s) for (ax, s), tok in _AXIS_TO_TOKENS.items()}


def _tokens_to_vec(tokens: set[str]) -> np.ndarray:
    """{action tokens} -> signed [x, y, z, yaw] vector ('hover' -> zeros)."""
    idx = {"x": 0, "y": 1, "z": 2, "yaw": 3}
    v = np.zeros(4)
    for t in tokens:
        if t in _TOKEN_TO_AXIS:
            ax, s = _TOKEN_TO_AXIS[t]
            v[idx[ax]] = s
    return v


def _vec_to_tokens(v: np.ndarray) -> set[str]:
    """Signed [x, y, z, yaw] -> action token set ('hover' if all zero)."""
    axes = ["x", "y", "z", "yaw"]
    toks = {_AXIS_TO_TOKENS[(ax, int(np.sign(val)))]
            for ax, val in zip(axes, v) if val != 0}
    return toks or {"hover"}


# --------------------------------------------------------------------------- #
# Per-frame features
# --------------------------------------------------------------------------- #
@dataclass
class FrameFeatures:
    frame_local: int
    n_tracks: int
    centroid_x: float           # normalized 0..1
    centroid_y: float
    mean_px: float              # mean longest-side over followed animals
    pct_vigilant: float
    majority_pose: str
    altitude: float


def _bbox_px(row) -> float:
    return float(max(row.w, row.h))


def _frame_features(g: pd.DataFrame, params: Params) -> FrameFeatures:
    """Compute frame-level aggregates from the visible tracks at one frame."""
    vis = g[~g["outside"].astype(str).str.lower().isin(["true", "1"])]
    n = len(vis)
    if n == 0:
        return FrameFeatures(int(g["frame_local"].iloc[0]), 0,
                             0.5, 0.5, 0.0, 0.0, "", float("nan"))
    # follow the N largest bboxes (subsumes fission/fusion: larger subgroup wins)
    vis = vis.assign(_area=vis["w"] * vis["h"]).nlargest(params.max_animals, "_area")
    cx = float(vis["x_c"].mean()) / FRAME_W
    cy = float(vis["y_c"].mean()) / FRAME_H
    mean_px = float(vis.apply(_bbox_px, axis=1).mean())
    vig = vis["vigilant"].astype(str).str.lower().isin(["true", "1"]).mean()
    poses = [p for p in vis["pose"].astype(str) if p and p != "nan"]
    majority = max(set(poses), key=poses.count) if poses else ""
    alt = float(vis["altitude"].iloc[0]) if "altitude" in vis else float("nan")
    return FrameFeatures(int(g["frame_local"].iloc[0]), int(n),
                         cx, cy, mean_px, float(vig), majority, alt)


# --------------------------------------------------------------------------- #
# Per-maneuver decision functions: features -> (token set, branch id)
# --------------------------------------------------------------------------- #
def decide_track(f: FrameFeatures, p: Params) -> tuple[set[str], str]:
    if f.n_tracks == 0:
        return {"hover"}, "no-detection"
    toks: set[str] = set()
    # horizontal recenter (keep herd centroid in center-50%)
    if f.centroid_x > KEEP_HI:
        toks.add("left")
    elif f.centroid_x < KEEP_LO:
        toks.add("right")
    # range control by apparent pixel size (X-Z plane only; no vertical)
    lo, hi = p.desired_pixels * (1 - p.pixel_band), p.desired_pixels * (1 + p.pixel_band)
    if f.mean_px < lo:
        toks.add("forward")
    elif f.mean_px > hi:
        toks.add("back")
    return (toks or {"hover"}), "track"


def decide_soi(f: FrameFeatures, p: Params) -> tuple[set[str], str]:
    if f.n_tracks == 0 or not f.majority_pose:
        return {"hover"}, "no-pose"
    if f.majority_pose not in POSE_RING or p.soi not in POSE_RING:
        return {"hover"}, "pose-unknown"
    if f.majority_pose != p.soi:
        # shortest signed rotation around the ring; +step == yaw-left
        i, j = POSE_RING.index(f.majority_pose), POSE_RING.index(p.soi)
        d = (j - i) % len(POSE_RING)
        step = d if d <= len(POSE_RING) - d else d - len(POSE_RING)
        return ({"yaw-left"} if step > 0 else {"yaw-right"}), "soi-rotate"
    # pose achieved -> close/retreat to desired pixels
    lo, hi = p.desired_pixels * (1 - p.pixel_band), p.desired_pixels * (1 + p.pixel_band)
    if f.mean_px < lo:
        return {"forward"}, "soi-range-in"
    if f.mean_px > hi:
        return {"back"}, "soi-range-out"
    return {"hover"}, "soi-hold"


def decide_approach(f: FrameFeatures, p: Params) -> tuple[set[str], str]:
    detected = (f.n_tracks > 0 and KEEP_LO <= f.centroid_x <= KEEP_HI
                and KEEP_LO <= f.centroid_y <= KEEP_HI)
    if detected:
        return {"hover"}, "approach-detected"   # handoff
    if not np.isnan(f.altitude):
        if f.altitude < p.launch_altitude - 1:
            return {"up"}, "approach-climb"
        if f.altitude > p.end_altitude + 1:
            return {"down"}, "approach-descend"
    return {"forward"}, "approach-search"


def decide_baf(series_St: np.ndarray, idx: int, p: Params) -> tuple[set[str], str]:
    """BAF is an override; evaluated on the smoothed S_t series with a hover hold."""
    if series_St[idx] >= p.theta_S:
        resp = {"back", "up"} if p.baf_response == "retreat" else {"hover"}
        return resp, "baf-trigger"
    # hover-hold: stay hovering for HOVER_HOLD frames after the last trigger
    lo = max(0, idx - HOVER_HOLD)
    if np.any(series_St[lo:idx] >= p.theta_S):
        return {"hover"}, "baf-hold"
    return set(), "baf-calm"     # empty -> no override


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def _smooth(raw_vecs: np.ndarray, window: int) -> np.ndarray:
    """Trailing rolling-mean each axis, then re-threshold past the dead-zone."""
    df = pd.DataFrame(raw_vecs, columns=["x", "y", "z", "yaw"])
    sm = df.rolling(window, min_periods=1).mean()
    out = np.where(sm.abs() <= AXIS_DEADZONE, 0, np.sign(sm))
    return out


# Maneuver-name aliases: the spec's suitability tags vs the harness branch names.
_MANEUVER_ALIASES = {"launch": "approach", "follow": "track"}


def _replay_table(df: pd.DataFrame, maneuver: str, params: Params,
                  smooth_window: int, clip_id: str = "") -> pd.DataFrame:
    """Core decision-tree replay over a frame-indexed labels DataFrame.

    Returns the long-format per-frame timeline (raw + smoothed action sets,
    triggering branch, and the frame-feature strips). `generate()` and
    `replay()` are thin wrappers that supply the DataFrame and window.
    """
    maneuver = _MANEUVER_ALIASES.get(maneuver, maneuver)
    frames = sorted(df["frame_local"].unique())
    feats = [_frame_features(df[df["frame_local"] == fl], params) for fl in frames]

    # BAF needs the smoothed S_t series first
    pct_vig = np.array([f.pct_vigilant for f in feats])
    St = pd.Series(pct_vig).rolling(VIGILANCE_WINDOW, min_periods=1).mean().to_numpy()

    rows, raw_vecs = [], []
    for i, f in enumerate(feats):
        if maneuver == "track":
            toks, branch = decide_track(f, params)
        elif maneuver == "soi_aware":
            toks, branch = decide_soi(f, params)
        elif maneuver == "approach":
            toks, branch = decide_approach(f, params)
        elif maneuver == "behavior_adaptive":
            toks, branch = decide_baf(St, i, params)
            toks = toks or {"hover"}
        else:
            raise ValueError(f"unknown maneuver {maneuver!r}")
        raw_vecs.append(_tokens_to_vec(toks))
        rows.append((f, branch, toks))

    smoothed = _smooth(np.array(raw_vecs), smooth_window)

    out = []
    for (f, branch, raw_toks), sv in zip(rows, smoothed):
        out.append({
            "clip_id": clip_id,
            "frame_local": f.frame_local,
            "maneuver": maneuver,
            "action_set_raw": "|".join(sorted(raw_toks)),
            "action_set_smoothed": "|".join(sorted(_vec_to_tokens(sv))),
            "triggering_branch": branch,
            "S_t": round(float(St[f.frame_local]) if f.frame_local < len(St) else 0.0, 3),
            "pct_vigilant": round(f.pct_vigilant, 3),
            "n_tracks": f.n_tracks,
            "centroid_x": round(f.centroid_x, 3),
            "centroid_y": round(f.centroid_y, 3),
            "mean_px": round(f.mean_px, 1),
        })
    return pd.DataFrame(out)


def generate(labels_csv: str, maneuver: str, params: Params) -> pd.DataFrame:
    """Replay one maneuver over a clip's ``labels.csv`` (canonical batch path).

    Uses the module's default smoothing window; clip_id is taken from the
    containing directory name (``clips/<clip_id>/labels.csv``).
    """
    df = pd.read_csv(labels_csv, low_memory=False)
    clip_id = os.path.basename(os.path.dirname(labels_csv))
    return _replay_table(df, maneuver, params, SMOOTH_WINDOW, clip_id)


def replay(labels, maneuver: str, thresholds: dict | None = None,
           params: Params | None = None) -> pd.DataFrame:
    """Replay a maneuver with reviewer-set thresholds (interactive entry point).

    Args:
        labels: a per-frame-per-track labels DataFrame (as loaded from a clip's
            ``labels.csv``), or a path to one.
        maneuver: ``approach`` | ``track`` | ``behavior_adaptive`` | ``soi_aware``
            (the ``launch`` / ``follow`` aliases are accepted too).
        thresholds: any subset of ``DEFAULT_THRESHOLDS`` keys
            (``theta_S``, ``desired_bbox_px``, ``soi_pose``, ``smoothing_window``);
            missing keys fall back to the defaults.
        params: an optional base ``Params`` to start from (the thresholds above
            are applied on top); defaults to ``Params()``.

    Returns the long-format per-frame timeline DataFrame: ``action_set_raw`` and
    ``action_set_smoothed`` (the timeline), ``triggering_branch`` (the branch
    track), and the frame-feature strips (``S_t``, ``pct_vigilant``, ``n_tracks``,
    ``centroid_x``, ``centroid_y``, ``mean_px``). Passing the default thresholds
    reproduces the released ``maneuver_labels.csv`` exactly (determinism check).
    """
    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    p = params or Params()
    p.theta_S = float(t["theta_S"])
    p.desired_pixels = float(t["desired_bbox_px"])
    p.soi = str(t["soi_pose"])
    window = int(t["smoothing_window"])

    if isinstance(labels, pd.DataFrame):
        df = labels
        clip_id = (str(df["clip_id"].iloc[0])
                   if "clip_id" in df.columns and len(df) else "")
    else:
        df = pd.read_csv(labels, low_memory=False)
        clip_id = os.path.basename(os.path.dirname(str(labels)))
    return _replay_table(df, maneuver, p, window, clip_id)


def _clip_dirs(out_root: str) -> list[str]:
    clips = os.path.join(out_root, "clips")
    return sorted(
        os.path.join(clips, d) for d in os.listdir(clips)
        if os.path.exists(os.path.join(clips, d, "labels.csv"))
    ) if os.path.isdir(clips) else []


def main():
    ap = argparse.ArgumentParser(description="Generate deterministic maneuver-action labels")
    ap.add_argument("--out", default=schema.OUTPUT_ROOT)
    ap.add_argument("--clip", default=None, help="clip_id (default: all extracted)")
    ap.add_argument("--maneuver", default=None,
                    choices=["approach", "track", "behavior_adaptive", "soi_aware"],
                    help="default: each clip's suitable_maneuvers")
    ap.add_argument("--all", action="store_true")
    # a few inline param overrides
    ap.add_argument("--desired-pixels", type=float, default=None)
    ap.add_argument("--theta-s", type=float, default=None)
    ap.add_argument("--soi", default=None)
    args = ap.parse_args()

    clip_index = io_paths.read_table(os.path.join(args.out, "catalog", "clip_index"))
    suitable = dict(zip(clip_index["clip_id"], clip_index["suitable_maneuvers"].fillna("")))
    # map spec name 'launch' tag -> 'approach' maneuver
    name_map = {"launch": "approach", "follow": "track",
                "behavior_adaptive": "behavior_adaptive", "soi_aware": "soi_aware"}

    dirs = _clip_dirs(args.out)
    if args.clip:
        dirs = [d for d in dirs if os.path.basename(d) == args.clip]
        if not dirs:
            raise SystemExit(f"no extracted clip {args.clip!r} (run extract_clips first)")

    # A "what-if" run (maneuver subset or any tuned param) must NOT clobber the
    # canonical, default-param maneuver_labels.csv -- it writes a sidecar instead.
    is_tweak = bool(args.maneuver or args.desired_pixels is not None
                    or args.theta_s is not None or args.soi is not None)
    out_name = "maneuver_labels.custom.csv" if is_tweak else "maneuver_labels.csv"
    if is_tweak:
        print(f"[tweak run] writing {out_name} (canonical maneuver_labels.csv untouched)")

    n_written = 0
    for d in dirs:
        cid = os.path.basename(d)
        tags = [name_map.get(t, t) for t in str(suitable.get(cid, "")).split("|") if t]
        maneuvers = [args.maneuver] if args.maneuver else tags
        parts = []
        for mv in maneuvers:
            p = Params()
            if args.desired_pixels is not None:
                p.desired_pixels = args.desired_pixels
            if args.theta_s is not None:
                p.theta_S = args.theta_s
            if args.soi is not None:
                p.soi = args.soi
            parts.append(generate(os.path.join(d, "labels.csv"), mv, p))
        if not parts:
            continue
        res = pd.concat(parts, ignore_index=True)
        res.to_csv(os.path.join(d, out_name), index=False)
        n_written += 1
        summ = res.groupby("maneuver")["action_set_smoothed"].agg(
            lambda s: s.value_counts().idxmax())
        print(f"  {cid}: {', '.join(f'{m}->{a}' for m, a in summ.items())}")
    print(f"\nwrote {out_name} for {n_written} clip(s)")


if __name__ == "__main__":
    main()
