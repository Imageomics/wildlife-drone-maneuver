#!/usr/bin/env bash
# Rebuild the KABR maneuver test-clip library end-to-end (ACSOS artifact, Mode B).
# Stages 1-6 + the deterministic maneuver-label harness. CPU only.
#
# Requires read access to the KABR raw videos (see clip_library/schema.py paths).
# To only (re)run the harness on an already-extracted dataset, run just the last
# step: `python -m clip_library.maneuver_labels --all`.
#
# Usage: ./run_pipeline.sh
set -euo pipefail
cd "$(dirname "$0")"

echo "== Stage 1: scan_sessions =="        ; python -m clip_library.scan_sessions
echo "== Stage 2: select_clips =="         ; python -m clip_library.select_clips
echo "== Stage 3: extract_clips =="        ; python -m clip_library.extract_clips
echo "== Stage 4: add_pose_labels =="      ; python -m clip_library.add_pose_labels
echo "== Stage 5: make_qa_overlays =="     ; python -m clip_library.make_qa_overlays --stratified
echo "== Stage 6: build_dataset_card =="   ; python -m clip_library.build_dataset_card
echo "== Harness: maneuver_labels =="      ; python -m clip_library.maneuver_labels --all

echo "Done. Dataset under the OUTPUT_ROOT in clip_library/schema.py"
