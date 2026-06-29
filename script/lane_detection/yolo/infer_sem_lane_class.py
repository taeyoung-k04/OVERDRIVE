#!/usr/bin/env python3
"""Run YOLO semantic segmentation and save classified lane-marking results."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from rules import IMAGE_SUFFIXES


CLASS_TO_ID = {
    "background": 0,
    "road": 1,
    "lane_left": 2,
    "lane_center": 3,
    "lane_right": 4,
    "stop_line": 5,
}

MASK_OUTPUTS = {
    "road_mask": CLASS_TO_ID["road"],
    "lane_left_mask": CLASS_TO_ID["lane_left"],
    "lane_center_mask": CLASS_TO_ID["lane_center"],
    "lane_right_mask": CLASS_TO_ID["lane_right"],
    "stop_line_mask": CLASS_TO_ID["stop_line"],
}

OVERLAY_COLORS = {
    CLASS_TO_ID["road"]: (35, 20, 0),
    CLASS_TO_ID["lane_left"]: (255, 80, 40),
    CLASS_TO_ID["lane_center"]: (0, 230, 255),
    CLASS_TO_ID["lane_right"]: (80, 255, 80),
    CLASS_TO_ID["stop_line"]: (0, 0, 255),
}


def semantic_to_class_map(semantic, shape: tuple[int, int]) -> np.ndarray:
    if semantic is None:
        return np.zeros(shape, dtype=np.uint8)

    class_map = semantic.data.cpu().numpy().astype(np.uint8)
    if class_map.shape != shape:
        class_map = cv2.resize(class_map, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return class_map


def make_class_overlay(image: np.ndarray, class_map: np.ndarray) -> np.ndarray:
    overlay = image.copy()
    tint = np.zeros_like(image)
    road_id = CLASS_TO_ID["road"]
    tint[class_map == road_id] = OVERLAY_COLORS[road_id]
    overlay = cv2.addWeighted(overlay, 1.0, tint, 0.24, 0)

    for class_name in ("lane_left", "lane_center", "lane_right", "stop_line"):
        class_id = CLASS_TO_ID[class_name]
        mask = (class_map == class_id).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, OVERLAY_COLORS[class_id], 5, cv2.LINE_AA)
        overlay[mask > 0] = OVERLAY_COLORS[class_id]
    return overlay


def save_prediction(image_path: Path, input_root: Path, output_root: Path, class_map: np.ndarray) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    relative = image_path.relative_to(input_root).with_suffix(".png")
    results = {
        "overlay": make_class_overlay(image, class_map),
        "class_map": class_map,
    }
    for directory, class_id in MASK_OUTPUTS.items():
        results[directory] = (class_map == class_id).astype(np.uint8) * 255

    for directory, result in results.items():
        destination = output_root / directory / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(destination), result):
            raise RuntimeError(f"Could not write image: {destination}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("dataset/lane_detection/frames"))
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("runs/semantic/yolo_sem_lane_class/train_cpu_640_yolo26n_ade20k/weights/best.pt"),
    )
    parser.add_argument("--output", type=Path, default=Path("result/lane_detection/yolo_sem_lane_class"))
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
        class_map = semantic_to_class_map(results[0].semantic_mask, image.shape[:2])
        save_prediction(source, args.input, args.output, class_map)
        print(f"[{index:>3}/{len(sources)}] {source}")

    print(f"Saved classified lane semantic results to {args.output}")


if __name__ == "__main__":
    main()
