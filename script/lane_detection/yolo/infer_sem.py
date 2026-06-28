#!/usr/bin/env python3
"""Run a trained YOLO semantic segmentation model and save road/lane masks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from rules import IMAGE_SUFFIXES, make_overlay


ROAD_ID = 1
LANE_ID = 2


def semantic_to_masks(class_map: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    road = (class_map == ROAD_ID).astype(np.uint8) * 255
    lane = (class_map == LANE_ID).astype(np.uint8) * 255
    return road, lane


def save_prediction(
    image_path: Path,
    input_root: Path,
    output_root: Path,
    road: np.ndarray,
    lane: np.ndarray,
) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    overlay = make_overlay(image, lane, road)
    relative = image_path.relative_to(input_root)
    for directory, result in {
        "overlay": overlay,
        "lane_mask": lane,
        "road_mask": road,
        "class_map": np.where(lane > 0, LANE_ID, np.where(road > 0, ROAD_ID, 0)).astype(np.uint8),
    }.items():
        destination = output_root / directory / relative.with_suffix(".png")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(destination), result):
            raise RuntimeError(f"Could not write image: {destination}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("dataset/lane_detection/frames"))
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("runs/semantic/runs/yolo_lane_sem/train_cpu_640_yolo26n_ade20k/weights/best.pt"),
    )
    parser.add_argument("--output", type=Path, default=Path("result/lane_detection/yolo_sem"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "ultralytics is not installed in the active environment. "
            "Install it in the overdrive conda env before inference."
        ) from exc

    model = YOLO(str(args.weights), task="semantic")
    sources = sorted(
        path for path in args.input.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not sources:
        raise SystemExit(f"No images found below {args.input}")

    for index, source in enumerate(sources, 1):
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read image: {source}")
        results = model.predict(
            source=image,
            imgsz=args.imgsz,
            device=args.device,
            task="semantic",
            verbose=False,
        )
        semantic = results[0].semantic_mask
        if semantic is None:
            road = np.zeros(image.shape[:2], dtype=np.uint8)
            lane = np.zeros(image.shape[:2], dtype=np.uint8)
        else:
            class_map = semantic.data.cpu().numpy().astype(np.uint8)
            if class_map.shape != image.shape[:2]:
                class_map = cv2.resize(
                    class_map,
                    (image.shape[1], image.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            road, lane = semantic_to_masks(class_map)
        save_prediction(source, args.input, args.output, road, lane)
        print(f"[{index:>3}/{len(sources)}] {source}")

    print(f"Saved YOLO semantic segmentation results to {args.output}")


if __name__ == "__main__":
    main()
