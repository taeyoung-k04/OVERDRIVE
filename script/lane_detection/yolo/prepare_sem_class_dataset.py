#!/usr/bin/env python3
"""Build a YOLO semantic dataset for classified lane markings."""

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


CLASS_NAMES = (
    "background",
    "road",
    "lane_left",
    "lane_center",
    "lane_right",
    "stop_line",
)
BACKGROUND_ID = 0
ROAD_ID = 1
LANE_LEFT_ID = 2
LANE_CENTER_ID = 3
LANE_RIGHT_ID = 4
STOP_LINE_ID = 5
IGNORE_ID = 255

LABEL_TO_ID = {
    "road": ROAD_ID,
    "lane_left": LANE_LEFT_ID,
    "lane_center": LANE_CENTER_ID,
    "lane_right": LANE_RIGHT_ID,
    "stop_line": STOP_LINE_ID,
}

PAINT_LABELS = ("lane_left", "lane_center", "lane_right", "stop_line")


def labeled_frames(label_root: Path) -> list[tuple[str, int]]:
    frames: set[tuple[str, int]] = set()
    for path in label_root.glob("*/*/*.png"):
        if path.name.endswith("~"):
            continue
        route, kind = path.relative_to(label_root).parts[:2]
        if kind in LABEL_TO_ID:
            frames.add((route, frame_second(path)))
    return sorted(frames)


def _read_optional_mask(label_root: Path, route: str, kind: str, second: int, shape: tuple[int, int]) -> np.ndarray | None:
    path = label_root / route / kind / f"frame_{second:06d}s.png"
    if not path.exists():
        return None
    return read_mask(path, shape)


def build_semantic_mask(
    label_root: Path,
    route: str,
    second: int,
    shape: tuple[int, int],
    ignore_unlabeled: bool,
    paint_margin: int,
) -> np.ndarray:
    fill_id = IGNORE_ID if ignore_unlabeled else BACKGROUND_ID
    semantic = np.full(shape, fill_id, dtype=np.uint8)

    road = _read_optional_mask(label_root, route, "road", second, shape)
    if road is not None:
        semantic[road > 0] = ROAD_ID

    paint_kernel = None
    if paint_margin > 0:
        paint_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * paint_margin + 1, 2 * paint_margin + 1),
        )

    # Later labels intentionally win overlaps. stop_line goes last because it is
    # a lane marking type, but should remain distinguishable where it crosses.
    for kind in PAINT_LABELS:
        mask = _read_optional_mask(label_root, route, kind, second, shape)
        if mask is None:
            continue
        if paint_kernel is not None:
            mask = cv2.dilate(mask, paint_kernel)
        semantic[mask > 0] = LABEL_TO_ID[kind]

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
    paint_margin: int,
    clean: bool,
) -> None:
    frames = labeled_frames(label_root)
    if not frames:
        raise SystemExit(f"No labels found below {label_root}")

    if clean and output.exists():
        shutil.rmtree(output)

    train_frames, val_frames = split_frames(frames, val_ratio)
    written = {"train": 0, "val": 0}
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
                paint_margin,
            )
            if not cv2.imwrite(str(mask_dir / f"{stem}.png"), semantic):
                raise RuntimeError(f"Could not write semantic mask for {stem}")
            written[split] += 1

    write_yaml(output)
    print(f"frames: {len(frames)}")
    print(f"train: {len(train_frames)} ({written['train']} written)")
    print(f"val: {len(val_frames)} ({written['val']} written)")
    print(f"ignore_unlabeled: {ignore_unlabeled}")
    print(f"paint_margin: {paint_margin}")
    print(f"saved: {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("dataset/lane_detection/frames"))
    parser.add_argument("--label", type=Path, default=Path("dataset/lane_detection/labels"))
    parser.add_argument("--output", type=Path, default=Path("dataset/lane_detection/yolo_lane_sem_class"))
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument(
        "--ignore-unlabeled",
        action="store_true",
        help="Mark unlabeled pixels as ignore instead of background.",
    )
    parser.add_argument("--paint-margin", type=int, default=1)
    parser.add_argument("--clean", action="store_true", help="Remove the output dataset before writing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_dataset(
        args.input,
        args.label,
        args.output,
        args.val_ratio,
        ignore_unlabeled=args.ignore_unlabeled,
        paint_margin=args.paint_margin,
        clean=args.clean,
    )


if __name__ == "__main__":
    main()
