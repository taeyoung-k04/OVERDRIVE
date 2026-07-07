#!/usr/bin/env python3
"""Run classified YOLO segmentation on lane-detection frames."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
CLASS_COLORS = {
    "car": (80, 80, 255),
    "road_left": (255, 130, 40),
    "road_right": (40, 210, 80),
    "stop_line": (0, 255, 255),
    "traffic_light": (255, 80, 255),
}
FALLBACK_COLORS = [
    (255, 80, 80),
    (0, 180, 255),
    (80, 220, 80),
    (0, 255, 255),
    (220, 80, 255),
]


def class_color(class_id: int, names: dict[int, str]) -> tuple[int, int, int]:
    return CLASS_COLORS.get(names.get(class_id, str(class_id)), FALLBACK_COLORS[class_id % len(FALLBACK_COLORS)])


def result_to_masks(result, image_shape: tuple[int, int], confidence: float) -> list[tuple[int, np.ndarray, float]]:
    if result.masks is None or result.boxes is None:
        return []

    height, width = image_shape
    raw_masks = result.masks.data.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    scores = result.boxes.conf.cpu().numpy()

    masks: list[tuple[int, np.ndarray, float]] = []
    for raw_mask, class_id, score in zip(raw_masks, classes, scores):
        if score < confidence:
            continue
        mask = (raw_mask > 0.5).astype(np.uint8) * 255
        if mask.shape != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        masks.append((int(class_id), mask, float(score)))
    return masks


def make_alpha_overlay(
    image: np.ndarray,
    masks: list[tuple[int, np.ndarray, float]],
    names: dict[int, str],
    alpha: float,
) -> np.ndarray:
    overlay = image.copy()
    tint = np.zeros_like(image)
    active = np.zeros(image.shape[:2], dtype=bool)

    for class_id, mask, _score in masks:
        pixels = mask > 0
        tint[pixels] = class_color(class_id, names)
        active |= pixels

    if np.any(active):
        blended = cv2.addWeighted(image, 1.0 - alpha, tint, alpha, 0.0)
        overlay[active] = blended[active]

    for class_id, mask, score in masks:
        color = class_color(class_id, names)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, 2, cv2.LINE_AA)
        if contours:
            contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(contour) >= 24:
                x, y, _w, _h = cv2.boundingRect(contour)
                cv2.putText(
                    overlay,
                    f"{names.get(class_id, str(class_id))} {score:.2f}",
                    (x, max(18, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA,
                )
    return overlay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("dataset/lane_detection/frames"))
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("runs/segment/yolo_lane_seg_class/train_cpu_960_yolo26n/weights/best.pt"),
    )
    parser.add_argument("--output", type=Path, default=Path("result/lane_detection/yolo_seg_class"))
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.12)
    parser.add_argument("--alpha", type=float, default=0.35)
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

    model = YOLO(str(args.weights.resolve()))
    sources = sorted(path for path in args.input.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if not sources:
        raise SystemExit(f"No images found below {args.input}")

    for index, source in enumerate(sources, 1):
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read image: {source}")
        result = model.predict(source=image, imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False)[0]
        names = getattr(result, "names", None) or getattr(model, "names", {})
        overlay = make_alpha_overlay(image, result_to_masks(result, image.shape[:2], args.conf), names, args.alpha)
        destination = args.output / "overlay" / source.relative_to(args.input).with_suffix(".png")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(destination), overlay):
            raise RuntimeError(f"Could not write image: {destination}")
        print(f"[{index:>3}/{len(sources)}] {source}", flush=True)

    print(f"Saved classified YOLO segmentation overlays to {args.output}")


if __name__ == "__main__":
    main()
