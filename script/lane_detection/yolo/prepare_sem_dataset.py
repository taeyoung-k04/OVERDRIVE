#!/usr/bin/env python3
"""Build a YOLO semantic segmentation dataset from road/lane label masks."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from label_guided import frame_second, read_mask
from prepare_seg_dataset import split_frames


CLASS_NAMES = ("background", "road", "lane")
BACKGROUND_ID = 0
ROAD_ID = 1
LANE_ID = 2
IGNORE_ID = 255


def labeled_frames(label_root: Path) -> list[tuple[str, int]]:
    frames: set[tuple[str, int]] = set()
    for path in label_root.glob("*/*/*.png"):
        if path.name.endswith("~"):
            continue
        route, kind = path.relative_to(label_root).parts[:2]
        if kind in {"road", "lane"}:
            frames.add((route, frame_second(path)))
    return sorted(frames)


def build_semantic_mask(
    label_root: Path,
    route: str,
    second: int,
    shape: tuple[int, int],
    ignore_unlabeled: bool,
    lane_margin: int,
) -> np.ndarray:
    if ignore_unlabeled:
        semantic = np.full(shape, IGNORE_ID, dtype=np.uint8)
    else:
        semantic = np.full(shape, BACKGROUND_ID, dtype=np.uint8)

    road_path = label_root / route / "road" / f"frame_{second:06d}s.png"
    lane_path = label_root / route / "lane" / f"frame_{second:06d}s.png"

    if road_path.exists():
        road = read_mask(road_path, shape) > 0
        semantic[road] = ROAD_ID

    if lane_path.exists():
        lane = read_mask(lane_path, shape)
        if lane_margin > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (2 * lane_margin + 1, 2 * lane_margin + 1),
            )
            lane = cv2.dilate(lane, kernel)
        semantic[lane > 0] = LANE_ID

    return semantic


def write_yaml(output: Path) -> None:
    names = "\n".join(f"  {index}: {name}" for index, name in enumerate(CLASS_NAMES))
    yaml = (
        f"path: {output.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "masks_dir: annotations\n"
        f"names:\n{names}\n"
    )
    (output / "data.yaml").write_text(yaml, encoding="utf-8")


def build_dataset(
    image_root: Path,
    label_root: Path,
    output: Path,
    val_ratio: float,
    ignore_unlabeled: bool,
    lane_margin: int,
) -> None:
    frames = labeled_frames(label_root)
    if not frames:
        raise SystemExit(f"No labels found below {label_root}")

    train_frames, val_frames = split_frames(frames, val_ratio)
    for split, split_frame_set in (("train", train_frames), ("val", val_frames)):
        image_dir = output / "images" / split
        mask_dir = output / "annotations" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        for route, second in sorted(split_frame_set):
            image_path = image_root / route / f"frame_{second:06d}s.jpg"
            if not image_path.exists():
                print(f"skip missing image: {image_path}")
                continue
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                print(f"skip unreadable image: {image_path}")
                continue

            stem = f"{route}__frame_{second:06d}s"
            shutil.copy2(image_path, image_dir / f"{stem}.jpg")
            semantic = build_semantic_mask(
                label_root,
                route,
                second,
                image.shape[:2],
                ignore_unlabeled,
                lane_margin,
            )
            if not cv2.imwrite(str(mask_dir / f"{stem}.png"), semantic):
                raise RuntimeError(f"Could not write semantic mask for {stem}")

    write_yaml(output)
    print(f"frames: {len(frames)}")
    print(f"train: {len(train_frames)}")
    print(f"val: {len(val_frames)}")
    print(f"ignore_unlabeled: {ignore_unlabeled}")
    print(f"lane_margin: {lane_margin}")
    print(f"saved: {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("dataset/lane_detection/frames"))
    parser.add_argument("--label", type=Path, default=Path("dataset/lane_detection/labels"))
    parser.add_argument("--output", type=Path, default=Path("dataset/lane_detection/yolo_lane_sem"))
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument(
        "--include-background",
        action="store_true",
        default=True,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--ignore-unlabeled",
        action="store_true",
        help="Mark unlabeled pixels as ignore instead of background.",
    )
    parser.add_argument("--lane-margin", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_dataset(
        args.input,
        args.label,
        args.output,
        args.val_ratio,
        ignore_unlabeled=args.ignore_unlabeled,
        lane_margin=args.lane_margin,
    )


if __name__ == "__main__":
    main()
