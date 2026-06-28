#!/usr/bin/env python3
"""Run a trained YOLO segmentation model and save road/lane masks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from rules import IMAGE_SUFFIXES, make_overlay


ROAD_CLASS_ID = 0
LANE_CLASS_ID = 1


def combine_masks(result, image_shape: tuple[int, int], confidence: float) -> tuple[np.ndarray, np.ndarray]:
    road = np.zeros(image_shape, dtype=np.uint8)
    lane = np.zeros(image_shape, dtype=np.uint8)
    if result.masks is None or result.boxes is None:
        return road, lane

    masks = result.masks.data.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    scores = result.boxes.conf.cpu().numpy()

    for mask, class_id, score in zip(masks, classes, scores):
        if score < confidence:
            continue
        mask_u8 = (mask > 0.5).astype(np.uint8) * 255
        if mask_u8.shape != image_shape:
            mask_u8 = cv2.resize(mask_u8, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)
        if class_id == ROAD_CLASS_ID:
            road = cv2.bitwise_or(road, mask_u8)
        elif class_id == LANE_CLASS_ID:
            lane = cv2.bitwise_or(lane, mask_u8)

    if np.any(road):
        margin = max(6, int(round(image_shape[1] * 0.004)))
        corridor = cv2.dilate(
            road,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1)),
        )
        lane = cv2.bitwise_and(lane, corridor)
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
        default=Path("runs/segment/runs/yolo_lane_seg/train_cpu_640_yolo26n/weights/best.pt"),
    )
    parser.add_argument("--output", type=Path, default=Path("result/lane_detection/yolo_seg"))
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.20)
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

    model = YOLO(str(args.weights))
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
            conf=args.conf,
            device=args.device,
            verbose=False,
        )
        road, lane = combine_masks(results[0], image.shape[:2], args.conf)
        save_prediction(source, args.input, args.output, road, lane)
        print(f"[{index:>3}/{len(sources)}] {source}")

    print(f"Saved YOLO segmentation results to {args.output}")


if __name__ == "__main__":
    main()
