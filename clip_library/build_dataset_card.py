"""Stage 6: generate DATASET_CARD.md following the Imageomics HF template.

Populates the template's sections from the catalog (composition, schema, the
three label provenances, limitations, license, citation) -- no "More
Information Needed" placeholders left for the parts we can fill.

Usage:
    python -m clip_library.build_dataset_card [--out DIR]
"""

from __future__ import annotations

import os
import argparse
import collections

import pandas as pd

from . import schema, io_paths

NSF_ACK = (
    "This work was supported by the [Imageomics Institute](https://imageomics.org), "
    "which is funded by the US National Science Foundation's Harnessing the Data "
    "Revolution (HDR) program under [Award #2118240](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2118240) "
    "(Imageomics: A New Frontier of Biological Information Powered by Knowledge-Guided "
    "Machine Learning). Any opinions, findings and conclusions or recommendations "
    "expressed in this material are those of the author(s) and do not necessarily "
    "reflect the views of the National Science Foundation."
)


def _counts(series_lists):
    c = collections.Counter()
    for v in series_lists:
        for tok in str(v).split("|"):
            if tok and tok.lower() != "nan":
                c[tok] += 1
    return c


def _size_cat(n: int) -> str:
    if n < 1000:
        return "n<1K"
    if n < 10000:
        return "1K<n<10K"
    return "10K<n<100K"


def build(out_root: str) -> str:
    ci = io_paths.read_table(os.path.join(out_root, "catalog", "clip_index"))
    n_clips = len(ci)
    n_videos = ci["video_id"].nunique()
    n_sessions = ci["session_id"].nunique()
    species = _counts(ci["species_set"])
    sizes = _counts(ci["bbox_size_classes"])
    mans = _counts(ci["suitable_maneuvers"])
    habitats = _counts(ci["habitat"])
    pose_clips = int((ci["pose_frames_covered"] > 0).sum())

    # pose audit (if present)
    pose_assigned = pose_ambig = 0
    audit_p = os.path.join(out_root, "catalog", "pose_audit")
    if os.path.exists(audit_p + ".csv"):
        adf = pd.read_csv(audit_p + ".csv")
        pose_assigned = int(adf["assigned"].sum())
        pose_ambig = int(adf["ambiguous"].sum())

    tags = (
        "biology, image, animals, CV, drone, UAV, KABR, MMLA, zebra, giraffe, "
        "Grevy's zebra, plains zebra, behavior, pose, animal-tracking, "
        "object-detection, video, Mpala Research Centre, Kenya"
    )
    species_lines = "\n".join(f"  - {k}: {v} clips" for k, v in species.most_common())
    man_lines = "\n".join(f"  - {k}: {v} clips" for k, v in mans.most_common())
    size_lines = "\n".join(f"  - {k}: {v} clips" for k, v in sizes.most_common())
    hab_lines = "\n".join(f"  - {k}: {v} clips" for k, v in habitats.most_common() if k != "nan")

    md = f"""---
license: cc-by-4.0
language:
- en
pretty_name: "KABR Maneuver Test-Clip Library"
task_categories:
- object-detection
- video-classification
tags:
- biology
- image
- animals
- CV
- drone
- UAV
- KABR
- zebra
- giraffe
- behavior
- pose
- animal-tracking
- Mpala Research Centre
size_categories:
- {_size_cat(n_clips)}
description: "A library of 6-second aerial drone clips derived from the KABR dataset (Mpala Research Centre, Kenya), each indexed by the autonomous-flight maneuver it is suitable for testing, with per-frame-per-track labels (bounding box, species, behaviour, persistent track id, ground-truth pose where available, telemetry)."
---

# Dataset Card for KABR Maneuver Test-Clip Library

A benchmark of **{n_clips} six-second drone clips** ({schema.CLIP_FRAMES} frames @ {schema.FPS} fps)
cut from {n_videos} KABR videos across {n_sessions} survey sessions at the Mpala Research Centre,
Kenya. Each clip is indexed by which autonomous-flight **maneuver** it can test
(launch / follow / behavior-adaptive / SoI-aware) and ships with a per-frame-per-track label table.
The library is intended for evaluating drone navigation policies against real wildlife footage.

## Dataset Details

### Dataset Description

- **Curated by:** Kline et al. (derived from the KABR dataset)
- **Language(s) (NLP):** en
- **Homepage:** https://github.com/Imageomics/wildwing
- **Repository:** autonomous_drone_simulator (`clip_library/`)
- **Paper:** ACSOS 2026 artifact (forthcoming); Journal of Field Robotics (in prep)
- **Related dataset:** [imageomics/KABR](https://huggingface.co/datasets/imageomics/KABR),
  [imageomics/KABR-poses](https://huggingface.co/datasets/imageomics/KABR-poses)

This dataset repackages the KABR aerial behavior dataset into short, mix-and-match clips that each
exercise a specific drone maneuver, so navigation policies can be benchmarked per-maneuver rather
than only end-to-end. Labels are aligned per frame per tracked individual.

### Supported Tasks and Leaderboards

Object detection / tracking, behaviour recognition, pose (viewpoint) estimation, and
**maneuver-conditioned navigation-policy evaluation** (the primary intended use).

## Dataset Structure

```
kabr_clips/
    catalog/
        video_index.csv      # per source video: frame summary, metadata, resolved raw video
        clip_index.csv       # one row per clip (the master index)
        coverage_report.md   # species x habitat x bbox-size x maneuver coverage
        pose_audit.csv       # per-video GT-pose assignment audit
    clips/
        <clip_id>/
            clip.mp4             # 6 s, {schema.CLIP_FRAMES} frames @ {schema.FPS} fps
            labels.csv           # one row per frame per track
            maneuver_labels.csv  # per-frame ground-truth drone action, per maneuver
    qa/                      # bbox+label overlays + contact sheets for manual review
    DATASET_CARD.md
```

`clip_id = <date>-<video>_<start_frame>` (e.g. `18_01_2023_session_7-DJI_0070_000360`).

### Data Fields

**clips/<clip_id>/labels.csv** (one row per frame per track):
- `clip_id`, `video_id`, `session_id`: identifiers; `session_id` is the KABR/FAIR² session event.
- `frame_global`: absolute frame in the source video; `frame_local`: 0-based within the clip; `time_s`.
- `track_id`: KABR mini-scene id, **persistent within the source video**.
- `species`, `behaviour`: KABR expert ground-truth labels; `vigilant`: behaviour in {{Head Up, Running, Trotting}}.
- `pose`, `pose_provenance`, `pose_match_score`: 8-class viewpoint where ground-truth exists (see Annotations); else empty.
- `individual_id`: empty — global re-identification is future work (no ground truth exists).
- `xtl,ytl,xbr,ybr`, `x_c,y_c,w,h`, `bbox_area_frac`, `bbox_size_class` (far/medium/close, relative to this survey).
- `occluded`, `outside`: KABR annotation flags; `latitude`, `longitude`, `altitude`: drone telemetry; `date_time`.

**catalog/clip_index.csv** (one row per clip): identifiers, frame range, `species_set`, `habitat`,
`habitat_notes`, `herd_size`, `behaviours_present`, `has_vigilance`, `bbox_size_classes`, `pose_set`,
`suitable_maneuvers`, and the FAIR² event IDs. `habitat` is a structural class
(`open`/`closed`/`mixed`/`unknown`) derived from the original free-text field metadata (Bitterlich
relascope scores + remarks), which is preserved verbatim in `habitat_notes`.

**clips/<clip_id>/maneuver_labels.csv** (one row per frame per maneuver) — the **ground-truth drone
action** produced by replaying the formal maneuver decision tree over `labels.csv`:
- `maneuver`: one of `approach` / `track` / `behavior_adaptive` / `soi_aware`.
- `action_set_raw`, `action_set_smoothed`: the per-frame action(s) from the 9-action space
  (`up, down, forward, back, left, right, yaw-left, yaw-right, hover`); `smoothed` is the published
  label after a 3 s rolling average that suppresses jitter.
- `triggering_branch`: which decision-tree branch fired (for auditability).
- `S_t`, `pct_vigilant`, `centroid_x`, `centroid_y`, `mean_px`, `n_tracks`: the frame features the
  decision used, so any label is reproducible and inspectable.

### Evaluation harness (maneuver decision tree)

The accompanying replay harness (`clip_library/maneuver_labels.py`, spec in
`maneuver_decision_tree.md`) executes a small, inspectable **policy specification** deterministically
over each clip, emitting the action an expert-calibrated controller would take per frame. A learned
navigation policy can then be scored, per maneuver, against this reference. Every threshold
(vigilance `theta_S`, desired bbox pixels, SoI pose, smoothing window) is user-tunable; tuned runs
write a `maneuver_labels.custom.csv` sidecar and never overwrite the released labels. The harness is
CPU-only and runs on the released clips without the raw KABR archive.

### Composition

- Clips: **{n_clips}** across **{n_videos}** videos / **{n_sessions}** sessions.
- Maneuver coverage:
{man_lines}
- Species (clip-level membership; `Zebra` is the coarse KABR label, see Annotations):
{species_lines}
- Bbox size class (relative to this survey's range):
{size_lines}
- Habitat structural class (original free text in `habitat_notes`):
{hab_lines}
- Clips with ground-truth pose: **{pose_clips}** (5 pose-annotated videos).

## Dataset Creation

### Curation Rationale

Prior drone-ecology evaluation is end-to-end and offline. This library isolates the *maneuvers* an
autonomous drone must perform (approaching, following a herd, responding to disturbance, capturing a
desired viewpoint) into short replayable clips with aligned ground truth, so navigation policies can
be benchmarked per maneuver under realistic perception conditions.

### Source Data

KABR full release (Mpala Research Centre drone surveys, January 2023; DJI aircraft, 20–50 m altitude).
Clips are cut from the original videos; labels are joined from the KABR per-frame occurrence records.
See KABR: https://doi.org/10.48550/arXiv.2510.02030

### Annotations — label provenance (read this)

Labels come from three distinct sources, stated explicitly:

1. **Bounding box, species, behaviour — KABR expert ground truth.** Frame-by-frame CVAT annotations
   from the KABR project; carried over verbatim (duplicate merge-rows collapsed to one row per track per frame).
2. **Pose (8-class viewpoint) — ground-truth cross-reference, sparse.** Sourced from the manually
   labeled `imageomics/KABR-poses` crops (5 videos: DJI_0002, DJI_0006, DJI_0070, DJI_0145, DJI_0208).
   We deliberately do **not** run the DINOv2 pose classifier here: it was trained on these same KABR
   crops, so applying it would be inference on training data. Because a crop's filename identifies its
   frame but not its track, each labeled crop is matched to a KABR track by **visual disambiguation**
   among the candidate boxes at that frame; matches below a similarity threshold are flagged
   `pose_provenance = "gt-ambiguous"`. Of {pose_assigned} crop assignments, {pose_ambig} were flagged
   ambiguous (see `catalog/pose_audit.csv`). Pose coverage is sparse by design.
3. **Individual ID — absent (future work).** No global re-identification ground truth exists; only the
   within-video `track_id` is provided. Cross-video re-ID (e.g. MegaDescriptor) is left for future work.

#### Who are the annotators?

KABR project annotators (bounding boxes, behaviour); the `imageomics/KABR-poses` curator (pose).
Maneuver-suitability tags are assigned programmatically by the selection pipeline.

### Personal and Sensitive Information

Wildlife only; no personal data. Species include Grevy's zebra (endangered) — coordinates are at
survey-session granularity.

## Considerations for Using the Data

### Bias, Risks, and Limitations

- **Pose is sparse** (5 videos) and best-effort track-matched; trust the `pose_provenance`/`pose_match_score` columns.
- **Species skew** toward Grevy's zebra; single site (Mpala); small bbox sizes throughout (20–50 m altitude),
  so "close/medium/far" are relative to this survey, not absolute scale.
- Labels are KABR ground truth, not model-generated — appropriate for ground-truth evaluation, not for
  characterizing perception error.
- Species/behaviour strings are normalized (`Walking`→`Walk`, `Grevy`→`Grevys Zebra`). The generic
  `Zebra` label is retained from KABR and means **undifferentiated zebra — Plains and/or Grevy's**:
  on those sessions the annotators did not split the two subspecies, so a `Zebra` clip may contain
  Plains zebra, Grevy's zebra, or both. A labeling-granularity difference, not noise.
- Because GT pose is sparse, the **SoI-aware** maneuver labels are mostly `hover` (no-pose) on this
  release; the maneuver is fully exercised only with dense (model-generated) pose.

### Recommendations

Filter on `pose_provenance == "gt"` for confident pose; use `suitable_maneuvers` to select clips per
maneuver; cross-reference `session_id` to the KABR/FAIR² metadata for habitat and herd context.

## Licensing Information

This dataset (the compilation) is released under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/).
Please also cite the original KABR and KABR-poses datasets.

## Citation

**BibTeX:**

**Data**
```
@misc{{kabr_maneuver_clips,
  author = {{Kline, Jenna and others}},
  title = {{KABR Maneuver Test-Clip Library}},
  year = {{2026}},
  url = {{https://huggingface.co/datasets/imageomics/kabr-maneuver-clips}},
  publisher = {{Hugging Face}}
}}
```

Please also cite the original data source(s):
```
@misc{{kabr2023,
  title  = {{KABR: In-Situ Dataset for Kenyan Animal Behavior Recognition from Drone Videos}},
  url    = {{https://doi.org/10.48550/arXiv.2510.02030}}
}}
```

## Acknowledgements

{NSF_ACK}

## Dataset Card Authors

Generated by the `clip_library` pipeline; curated by Kline et al.

## Dataset Card Contact

See the project repository.
"""
    return md


def main():
    ap = argparse.ArgumentParser(description="Stage 6: build dataset card")
    ap.add_argument("--out", default=schema.OUTPUT_ROOT)
    args = ap.parse_args()

    md = build(args.out)
    path = os.path.join(args.out, "DATASET_CARD.md")
    os.makedirs(args.out, exist_ok=True)
    with open(path, "w") as fh:
        fh.write(md)
    print("wrote", path, f"({len(md)} chars)")


if __name__ == "__main__":
    main()
