#!/usr/bin/env python3
"""Build a semantic-segmentation dataset from parking perspective images."""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import cv2
import numpy as np


CLASS_NAMES = ("background", "road", "park", "grass", "lane")
LABEL_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES) if index}
PAINT_ORDER = ("road", "park", "grass", "lane")
FRAME_RE = re.compile(r"frame_(\d+)s$")


def frame_second(path: Path) -> int:
    match = FRAME_RE.fullmatch(path.stem)
    if match is None:
        raise ValueError(f"Could not parse frame second from {path}")
    return int(match.group(1))


def read_binary_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise RuntimeError(f"Could not read mask: {path}")
    if mask.ndim == 3 and mask.shape[2] == 4:
        binary = mask[..., 3] > 8
    elif mask.ndim == 3:
        binary = np.any(mask[..., :3] > 8, axis=2)
    else:
        binary = mask > 8
    if binary.shape != shape:
        raise ValueError(
            f"Mask and perspective image sizes differ: {path} is "
            f"{binary.shape[::-1]}, expected {shape[::-1]}"
        )
    return binary


def labeled_frames(label_root: Path) -> list[tuple[str, int]]:
    frames: set[tuple[str, int]] = set()
    for path in label_root.glob("*/*/*.png"):
        relative = path.relative_to(label_root)
        route, label = relative.parts[:2]
        if label in LABEL_TO_ID:
            frames.add((route, frame_second(path)))
    return sorted(frames)


def split_frames(
    frames: list[tuple[str, int]], val_ratio: float
) -> tuple[set[tuple[str, int]], set[tuple[str, int]]]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("--val-ratio must be between 0 and 1")
    train: set[tuple[str, int]] = set()
    val: set[tuple[str, int]] = set()
    routes: dict[str, list[tuple[str, int]]] = {}
    for frame in frames:
        routes.setdefault(frame[0], []).append(frame)
    interval = max(2, round(1.0 / val_ratio))
    for route_frames in routes.values():
        for index, frame in enumerate(sorted(route_frames, key=lambda item: item[1])):
            (val if index % interval == interval - 1 else train).add(frame)
    if not val and len(train) > 1:
        held_out = sorted(train)[-1]
        train.remove(held_out)
        val.add(held_out)
    return train, val


def build_class_map(
    label_root: Path, route: str, second: int, shape: tuple[int, int]
) -> np.ndarray:
    class_map = np.zeros(shape, dtype=np.uint8)
    filename = f"frame_{second:06d}s.png"
    found = False
    for label in PAINT_ORDER:
        path = label_root / route / label / filename
        if not path.exists():
            continue
        class_map[read_binary_mask(path, shape)] = LABEL_TO_ID[label]
        found = True
    if not found:
        raise FileNotFoundError(f"No masks found for {route}/{filename}")
    return class_map


def write_yaml(output: Path) -> None:
    names = "\n".join(f"  {index}: {name}" for index, name in enumerate(CLASS_NAMES))
    content = (
        f"path: {output.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "masks_dir: annotations\n"
        f"names:\n{names}\n"
    )
    (output / "data.yaml").write_text(content, encoding="utf-8")


def build_dataset(
    image_root: Path,
    label_root: Path,
    output: Path,
    val_ratio: float,
    clean: bool,
) -> None:
    frames = labeled_frames(label_root)
    if not frames:
        raise SystemExit(f"No labels found below {label_root}")
    if clean and output.exists():
        shutil.rmtree(output)

    train, val = split_frames(frames, val_ratio)
    written = {"train": 0, "val": 0}
    for split, selected in (("train", train), ("val", val)):
        image_dir = output / "images" / split
        mask_dir = output / "annotations" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        for route, second in sorted(selected):
            source = image_root / route / f"frame_{second:06d}s.jpg"
            image = cv2.imread(str(source), cv2.IMREAD_COLOR)
            if image is None:
                print(f"skip missing or unreadable image: {source}")
                continue
            stem = f"{route}__frame_{second:06d}s"
            shutil.copy2(source, image_dir / f"{stem}.jpg")
            class_map = build_class_map(
                label_root, route, second, image.shape[:2]
            )
            destination = mask_dir / f"{stem}.png"
            if not cv2.imwrite(str(destination), class_map):
                raise RuntimeError(f"Could not write mask: {destination}")
            written[split] += 1

    write_yaml(output)
    print(f"labeled frames: {len(frames)}")
    print(f"train: {written['train']}, val: {written['val']}")
    print(f"saved: {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("dataset/parking/perspective"))
    parser.add_argument("--label", type=Path, default=Path("dataset/parking/labels"))
    parser.add_argument("--output", type=Path, default=Path("dataset/parking/yolo_parking_sem_class"))
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--clean", action="store_true", help="Remove the output before writing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_dataset(args.input, args.label, args.output, args.val_ratio, args.clean)


if __name__ == "__main__":
    main()
