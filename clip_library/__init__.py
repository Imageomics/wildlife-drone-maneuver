"""Drone-maneuver test-clip library — dataset build pipeline + replay harness.

A benchmark of 6-second, maneuver-indexed aerial drone clips (initially sourced
from the KABR wildlife dataset; more sources to come), each carrying
per-frame-per-track labels, plus a small deterministic policy that replays a
formal maneuver decision tree over the clips to emit ground-truth drone actions.

Two usage modes:

* **Mode A — run the harness on the released dataset (CPU-only, no raw video).**
  The common path. Needs only ``pandas`` / ``numpy``::

      python -m clip_library.maneuver_labels --all      # regenerate GT actions

  or, programmatically, ``maneuver_labels.generate`` / ``maneuver_labels.replay``.

* **Mode B — rebuild the clips from raw KABR video.** Needs the raw archive
  (not redistributed) and the ``[pipeline]`` extra (OpenCV). Stages, in order:
    1. scan_sessions     -> catalog/video_index
    2. select_clips      -> catalog/clip_index + coverage_report
    3. extract_clips     -> clips/<id>/clip.mp4 + labels
    4. add_pose_labels   -> GT pose cross-reference (5 videos)
    5. make_qa_overlays  -> qa/ overlays + contact sheets
    6. build_dataset_card-> dataset card

Importing this package is lightweight (no OpenCV); heavy deps are imported lazily
by the modules that need them.
"""

__all__ = ["schema", "io_paths", "maneuver_labels"]
