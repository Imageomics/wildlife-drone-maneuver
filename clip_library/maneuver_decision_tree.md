# 🚁 Maneuver Decision Tree — formal specification

> This document is both **(a)** the citable policy specification and **(b)** the exact
> logic that [`maneuver_labels.py`](maneuver_labels.py) executes deterministically over
> each clip's `labels.csv`. Every maneuver is a small table — **one row per branch:
> `condition → action`** — so any published label can be traced back to the rule that
> produced it.

**Contents**

- [0 · Action space & output model](#0--action-space--output-model)
- [1 · Per-frame features](#1--per-frame-features)
- [2 · The four maneuvers](#2--the-four-maneuvers)
  - [APPROACH](#maneuver-1--approach) · [TRACK](#maneuver-2--track) ·
    [BAF](#maneuver-3--behavior-adaptive-flight-baf) · [SoI-aware](#maneuver-4--soi-aware)
- [3 · Composition (mission)](#3--composition-mission)
- [4 · Generator output](#4--generator-output)
- [5 · Data hygiene](#5--data-hygiene)

#### The four maneuvers at a glance

| # | Maneuver | One-line objective | Key user params |
|---|----------|--------------------|-----------------|
| 1 | **APPROACH** | Begin the mission and position the drone without spooking wildlife | `launch_altitude`, `end_altitude`, `target_species` |
| 2 | **TRACK** | Keep the herd centroid centered at a desired apparent size | `desired_pixels`, `pixel_band`, `max_animals` |
| 3 | **BAF** | Detect disturbance and respond (override) | `theta_S`, `baf_response` |
| 4 | **SoI-aware** | Rotate to a desired viewpoint, then hold at a target size | `soi`, `desired_pixels` |

---

## 0 · Action space & output model

**Nine actions:** `up`, `down`, `forward`, `back`, `left`, `right`, `yaw-left`, `yaw-right`, `hover`.

- Per frame the generator emits a **set** of actions (e.g. `{forward, yaw-left}`);
  `hover` is its own explicit token, emitted when the set is otherwise empty.
- The raw per-frame sets are **smoothed** into the published action. Each set is
  decomposed onto four signed axes —

  | axis | −1 | +1 |
  |------|----|----|
  | `x`   | `left`     | `right`     |
  | `y`   | `down`     | `up`        |
  | `z`   | `back`     | `forward`   |
  | `yaw` | `yaw-left` | `yaw-right` |

  — averaged with a trailing window of **`W = 90` frames** (3 s @ 30 fps), then
  re-thresholded. An axis whose smoothed magnitude is within the dead-zone
  (`|mean| ≤ 0.33`) emits no motion; all axes quiet → `hover`.

---

## 1 · Per-frame features

> **Keep-in zone** — the center 50% of the frame: trim ¼ off each edge →
> `x ∈ [0.25, 0.75]`, `y ∈ [0.25, 0.75]` (frame 3840×2160). The herd is "centered"
> when its centroid is inside this box; corrections fire when it leaves.

### Per-track (from `labels.csv`)

| field | values (normalized) |
|---|---|
| `species` | Giraffe, Plains Zebra, Grevys Zebra (collapse `Grevy`→`Grevys Zebra`) |
| `behaviour` | Head Up, Walk (collapse `Walking`→`Walk`), Graze, Browsing, Running, Trotting, Auto-Groom |
| `vigilant` | `True` if behaviour ∈ {Head Up, Running, Trotting} |
| `pose` | front, front-left, front-right, left, right, back-left, back-right, back |
| bbox | `x_c, y_c`, `w, h`, `bbox_area_frac`, `bbox_size_class` ∈ {far, medium, close} |
| telemetry | `latitude, longitude, altitude` (used by APPROACH only) |

### Frame-level aggregates (derived)

`n_tracks`, `centroid` (mean `x_c, y_c` over the followed animals, normalized),
`mean_px` (mean longest bbox side over followed animals), `pct_vigilant`,
`S_t` (trailing 90-frame mean of `pct_vigilant`), `majority_pose`.

> When `n_tracks` exceeds `max_animals`, only the `max_animals` largest bboxes are
> followed (this also subsumes herd fission/fusion: the larger subgroup wins).

---

## 2 · The four maneuvers

### Maneuver 1 — APPROACH

> **Objective:** begin the mission and position the drone without spooking wildlife.
> **User params:** `launch_altitude` (50 m), `end_altitude` (30 m), `target_species`.

| # | condition | action | note |
|---|---|---|---|
| 1 | target detected, centroid in keep-zone | `hover` | approach complete → handoff |
| 2 | altitude < launch_altitude − 1 | `up` | climb to launch altitude |
| 3 | altitude > end_altitude + 1 | `down` | descend toward end altitude |
| 4 | otherwise (target not yet detected) | `forward` | search forward |

### Maneuver 2 — TRACK

> **Objective:** keep the majority of the herd centroid in the center-50% keep-zone,
> at a desired apparent size. Range control acts in the X–Z plane only (no vertical).
> **User params:** `desired_pixels` (30), `pixel_band` (±0.25), `max_animals` (5).

| # | condition | action | note |
|---|---|---|---|
| 1 | centroid.x right of keep-zone | `left` | recenter |
| 2 | centroid.x left of keep-zone | `right` | recenter |
| 3 | mean_px below `desired_pixels·(1−band)` | `forward` | too small → close in |
| 4 | mean_px above `desired_pixels·(1+band)` | `back` | too large → back off |
| – | no detection | `hover` | |

### Maneuver 3 — BEHAVIOR-ADAPTIVE FLIGHT (BAF)

> **Objective:** detect and respond to disturbance. Evaluated as an **override** on the
> smoothed vigilance series `S_t`, with a hover hold after each trigger.
> **User params:** `theta_S` (0.5), `baf_response` ∈ {retreat, hover}.

| # | condition | action | note |
|---|---|---|---|
| 1 | `S_t` ≥ theta_S | `retreat` = `{back, up}` (or `{hover}`) | override active maneuver |
| 2 | a trigger fired within the last 150 frames (5 s) | `hover` | hysteresis hold |
| 3 | otherwise | (no override; defer to active maneuver) | calm |

### Maneuver 4 — SoI-AWARE

> **Objective:** maneuver to capture the desired Surface of Interest (pose) of the
> majority of the herd, then hold at a target apparent size. Two stages.
> **User params:** `soi` (desired pose, default `left`), `desired_pixels`
> (per objective: track 30, behavior 100, re-ID 500).

The eight poses form a **ring**; one `yaw-left` step rotates apparent pose one
position around it (front → front-right → right → …). To reach the desired pose
the drone yaws the short way; `yaw-right` rotates the opposite direction.

| # | stage | condition | action |
|---|---|---|---|
| 1 | rotate | majority_pose ≠ soi | `yaw-left` / `yaw-right` (short way around ring) |
| 2 | range | majority_pose = soi, mean_px < target | `forward` |
| 3 | range | majority_pose = soi, mean_px > target | `back` |
| 4 | hold | majority_pose = soi, mean_px in band | `hover` |
| – | — | no pose available | `hover` |

---

## 3 · Composition (mission)

A mission may run several maneuvers; the intended override priority is

> **BAF → APPROACH** (until handoff) **→ TRACK → SoI**

This release generates labels per `(clip × maneuver × param-set)` **independently**;
cross-maneuver composition is left to the consumer.

---

## 4 · Generator output

Per `(clip × maneuver × param-set)`, long-format `maneuver_labels.csv` columns:

```
clip_id, frame_local, maneuver, action_set_raw, action_set_smoothed,
triggering_branch, S_t, pct_vigilant, n_tracks, centroid_x, centroid_y, mean_px
```

---

## 5 · Data hygiene

- Collapse `Walking`→`Walk`, `Grevy`→`Grevys Zebra` (applied at label-load and
  re-emitted in the released dataset).
- `vigilant` = behaviour ∈ {Head Up, Running, Trotting}.
