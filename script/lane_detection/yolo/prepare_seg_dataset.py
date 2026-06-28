#!/usr/bin/env python3
"""Build a YOLO segmentation dataset from lane-detection label masks."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from label_guided import frame_second, read_mask


CLASS_NAMES = ("road", "lane")
CLASS_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}


def mask_to_yolo_segments(
    mask: np.ndarray,
    class_id: int,
    min_area: int,
    epsilon_ratio: float,
) -> list[str]:
    height, width = mask.shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines: list[str] = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        epsilon = max(1.0, cv2.arcLength(contour, True) * epsilon_ratio)
        polygon = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
        if len(polygon) < 3:
            continue

        normalized: list[str] = []
        for x, y in polygon:
            normalized.append(f"{min(max(float(x) / width, 0.0), 1.0):.6f}")
            normalized.append(f"{min(max(float(y) / height, 0.0), 1.0):.6f}")
        lines.append(f"{class_id} {' '.join(normalized)}")

    return lines


def labeled_frames(label_root: Path) -> list[tuple[str, int]]:
    frames: set[tuple[str, int]] = set()
    for path in label_root.glob("*/*/*.png"):
        if path.name.endswith("~"):
            continue
        route, kind = path.relative_to(label_root).parts[:2]
        if kind in CLASS_TO_ID:
            frames.add((route, frame_second(path)))
    return sorted(frames)


def split_frames(frames: list[tuple[str, int]], val_ratio: float) -> tuple[set[tuple[str, int]], set[tuple[str, int]]]:
    train: set[tuple[str, int]] = set()
    val: set[tuple[str, int]] = set()

    by_route: dict[str, list[tuple[str, int]]] = {}
    for frame in frames:
        by_route.setdefault(frame[0], []).append(frame)

    for route_frames in by_route.values():
        route_frames = sorted(route_frames, key=lambda item: item[1])
        if len(route_frames) == 1:
            train.add(route_frames[0])
            continue
        interval = max(2, round(1.0 / max(val_ratio, 0.01)))
        for index, frame in enumerate(route_frames):
            if index % interval == interval - 1:
                val.add(frame)
            else:
                train.add(frame)

    if not val and train:
        frame = sorted(train)[-1]
        train.remove(frame)
        val.add(frame)
    return train, val


def write_yaml(output: Path) -> None:
    yaml = (
        f"path: {output.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: road\n"
        "  1: lane\n"
    )
    (output / "data.yaml").write_text(yaml, encoding="utf-8")


def build_dataset(
    image_root: Path,
    label_root: Path,
    output: Path,
    val_ratio: float,
    min_area: int,
    epsilon_ratio: float,
) -> None:
    frames = labeled_frames(label_root)
    if not frames:
        raise SystemExit(f"No labels found below {label_root}")

    train_frames, val_frames = split_frames(frames, val_ratio)

    for split, split_frames_set in (("train", train_frames), ("val", val_frames)):
        image_dir = output / "images" / split
        label_dir = output / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        for route, second in sorted(split_frames_set):
            image_path = image_root / route / f"frame_{second:06d}s.jpg"
            if not image_path.exists():
                print(f"skip missing image: {image_path}")
                continue

            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                print(f"skip unreadable image: {image_path}")
                continue

            stem = f"{route}__frame_{second:06d}s"
            destination_image = image_dir / f"{stem}.jpg"
            shutil.copy2(image_path, destination_image)

            lines: list[str] = []
            for class_name, class_id in CLASS_TO_ID.items():
                mask_path = label_root / route / class_name / f"frame_{second:06d}s.png"
                if not mask_path.exists():
                    continue
                mask = read_mask(mask_path, image.shape[:2])
                lines.extend(mask_to_yolo_segments(mask, class_id, min_area, epsilon_ratio))

            (label_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    write_yaml(output)
    print(f"frames: {len(frames)}")
    print(f"train: {len(train_frames)}")
    print(f"val: {len(val_frames)}")
    print(f"saved: {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("dataset/lane_detection/frames"))
    parser.add_argument("--label", type=Path, default=Path("dataset/lane_detection/labels"))
    parser.add_argument("--output", type=Path, default=Path("dataset/lane_detection/yolo_lane_seg"))
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--min-area", type=int, default=80)
    parser.add_argument("--epsilon-ratio", type=float, default=0.0025)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_dataset(
        args.input,
        args.label,
        args.output,
        args.val_ratio,
        args.min_area,
        args.epsilon_ratio,
    )


if __name__ == "__main__":
    main()
