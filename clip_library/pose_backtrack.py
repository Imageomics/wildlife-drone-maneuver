#!/usr/bin/env python3
"""
Backtrack a labeled pose crop to its source video, frame, and bounding box.

Pose crops are named by the KABR detection pipeline like:

    mpala_session_1_DJI_0002_partition_1_DJI_0002_000171_c0_004.jpg
    └────── session ──────┘ └─video─┘ └partition┘ └───frame jpg───┘ │   └ detection index
                                                    DJI_0002_000171   └ class id

This module gives you a single function, `backtrack(crop_filename, tracks_xml)`,
that parses the filename and resolves the bounding box from a CVAT-style
`*_tracks.xml` (absolute pixel coords) by frame number.

Note: the crop's `cN_NNN` suffix is a class id + per-frame detection index from
the YOLO partition pipeline, which does NOT map 1:1 onto CVAT track ids. So we
return *every* box present at that frame and let the caller disambiguate
(e.g. by visual match). The parsed metadata is exact; the bbox match is by frame.

Vendored verbatim from the project owner's helper script (see notes/060225.md).
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

# mpala_session_1 _ DJI_0002 _ partition_1 _ DJI_0002 _ 000171 _ c0 _ 004 .jpg
_CROP_RE = re.compile(
    r"^(?P<session>.+?)"
    r"_(?P<video>DJI_\d+)"
    r"_partition_(?P<partition>\d+)"
    r"_(?P=video)_(?P<frame>\d+)"
    r"_c(?P<class_id>\d+)"
    r"_(?P<det_index>\d+)"
    r"\.(?:jpg|jpeg|png)$",
    re.IGNORECASE,
)


@dataclass
class CropMeta:
    """Provenance parsed directly from the crop filename (exact)."""
    session: str          # e.g. "mpala_session_1"
    video: str            # e.g. "DJI_0002"
    partition: int        # e.g. 1
    frame: int            # absolute video frame index, e.g. 171
    class_id: int         # YOLO class id from the detection pipeline
    det_index: int        # per-frame detection index


@dataclass
class Box:
    """A bounding box recovered from the CVAT tracks XML at the crop's frame."""
    track_id: int
    label: str            # e.g. "Zebra", "Giraffe"
    x1: float             # xtl, pixels
    y1: float             # ytl
    x2: float             # xbr
    y2: float             # ybr
    occluded: bool

    def as_xyxy(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


def parse_crop_name(crop_filename: str) -> CropMeta:
    """Parse a crop filename into its provenance fields. Raises ValueError if it
    doesn't match the expected pattern."""
    name = Path(crop_filename).name
    m = _CROP_RE.match(name)
    if not m:
        raise ValueError(
            f"Crop filename {name!r} does not match the expected pattern "
            "'<session>_<video>_partition_<n>_<video>_<frame>_c<class>_<index>.<ext>'"
        )
    return CropMeta(
        session=m.group("session"),
        video=m.group("video"),
        partition=int(m.group("partition")),
        frame=int(m.group("frame")),
        class_id=int(m.group("class_id")),
        det_index=int(m.group("det_index")),
    )


def boxes_at_frame(tracks_xml: str | Path, frame: int) -> list[Box]:
    """Return all visible (outside=0) boxes at `frame` from a CVAT tracks XML."""
    root = ET.parse(str(tracks_xml)).getroot()
    boxes: list[Box] = []
    for track in root.findall("track"):
        track_id = int(track.get("id", -1))
        label = track.get("label", "")
        for box in track.findall("box"):
            if int(box.get("frame", -1)) != frame:
                continue
            if box.get("outside") == "1":
                continue
            boxes.append(
                Box(
                    track_id=track_id,
                    label=label,
                    x1=float(box.get("xtl")),
                    y1=float(box.get("ytl")),
                    x2=float(box.get("xbr")),
                    y2=float(box.get("ybr")),
                    occluded=box.get("occluded") == "1",
                )
            )
    return boxes


def backtrack(crop_filename: str, tracks_xml: str | Path) -> tuple[CropMeta, list[Box]]:
    """Backtrack a crop to (parsed provenance, candidate boxes at that frame).

    The CropMeta is exact. The box list is every annotated box at that frame;
    pick the one matching your crop (usually by visual overlap / count).
    """
    meta = parse_crop_name(crop_filename)
    boxes = boxes_at_frame(tracks_xml, meta.frame)
    return meta, boxes


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Backtrack a pose crop to video/frame/bbox")
    ap.add_argument("crop", help="Crop filename (or path); only the basename is parsed")
    ap.add_argument("--tracks", help="Path to the matching *_tracks.xml", default=None)
    args = ap.parse_args()

    meta = parse_crop_name(args.crop)
    print(f"session   : {meta.session}")
    print(f"video     : {meta.video}.mp4")
    print(f"partition : {meta.partition}")
    print(f"frame     : {meta.frame}")
    print(f"class_id  : {meta.class_id}")
    print(f"det_index : {meta.det_index}")

    if args.tracks:
        boxes = boxes_at_frame(args.tracks, meta.frame)
        print(f"\n{len(boxes)} box(es) at frame {meta.frame} in {Path(args.tracks).name}:")
        for b in boxes:
            print(
                f"  track {b.track_id:>3} {b.label:<10} "
                f"xyxy=({b.x1:.0f}, {b.y1:.0f}, {b.x2:.0f}, {b.y2:.0f})"
                f"{' [occluded]' if b.occluded else ''}"
            )
