#!/usr/bin/env python3
"""Run YOLO semantic segmentation and save lane/object class results."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from utils.perspective import (
    add_perspective_args,
    apply_perspective,
    make_perspective_config,
)
from utils.postprocess import (
    CLASS_TO_ID,
    add_postprocess_args,
    postprocess_class_map,
)


DEFAULT_WEIGHTS = Path("runs/semantic/yolo_lane_sem_class/train_cpu_640_yolo26n_8class/weights/best.pt")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}

MASK_OUTPUTS = {
    "road_mask": CLASS_TO_ID["road"],
    "lane_left_mask": CLASS_TO_ID["lane_left"],
    "lane_center_mask": CLASS_TO_ID["lane_center"],
    "lane_right_mask": CLASS_TO_ID["lane_right"],
    "stop_line_mask": CLASS_TO_ID["stop_line"],
    "car_mask": CLASS_TO_ID["car"],
    "traffic_light_mask": CLASS_TO_ID["traffic_light"],
}

OVERLAY_COLORS = {
    CLASS_TO_ID["road"]: (35, 20, 0),
    CLASS_TO_ID["lane_left"]: (255, 80, 40),
    CLASS_TO_ID["lane_center"]: (0, 230, 255),
    CLASS_TO_ID["lane_right"]: (80, 255, 80),
    CLASS_TO_ID["stop_line"]: (0, 0, 255),
    CLASS_TO_ID["car"]: (255, 0, 255),
    CLASS_TO_ID["traffic_light"]: (0, 165, 255),
}
OVERLAY_ALPHA = 0.32
OVERLAY_CLASS_NAMES = (
    "road",
    "lane_left",
    "lane_center",
    "lane_right",
    "stop_line",
    "car",
    "traffic_light",
)


def resolve_model_path(weights: Path, backend: str) -> Path:
    if backend == "auto":
        model_path = weights
    elif backend == "pt":
        model_path = weights.with_suffix(".pt") if weights.suffix.lower() != ".pt" else weights
    elif backend == "onnx":
        model_path = weights.with_suffix(".onnx") if weights.suffix.lower() != ".onnx" else weights
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    if not model_path.exists():
        raise SystemExit(
            f"Model file does not exist: {model_path}\n"
            "Use --weights to point to an existing .pt/.onnx file, or export ONNX first."
        )
    return model_path


def load_semantic_model(weights: Path, backend: str):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "ultralytics is not installed in the active environment. "
            "Install it in the overdrive conda env before inference."
        ) from exc

    model_path = resolve_model_path(weights, backend)
    print(f"Using {model_path.suffix.lower()[1:]} model: {model_path}")
    return YOLO(str(model_path), task="semantic")


def semantic_to_class_map(semantic, shape: tuple[int, int]) -> np.ndarray:
    if semantic is None:
        return np.zeros(shape, dtype=np.uint8)

    class_map = semantic.data.cpu().numpy().astype(np.uint8)
    if class_map.shape != shape:
        class_map = cv2.resize(class_map, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return class_map


def make_class_overlay(image: np.ndarray, class_map: np.ndarray) -> np.ndarray:
    overlay = image.copy()
    for class_name in OVERLAY_CLASS_NAMES:
        class_id = CLASS_TO_ID[class_name]
        mask = (class_map == class_id).astype(np.uint8) * 255
        if not np.any(mask):
            continue

        color = np.array(OVERLAY_COLORS[class_id], dtype=np.float32)
        mask_bool = mask > 0
        overlay[mask_bool] = (
            image[mask_bool].astype(np.float32) * (1.0 - OVERLAY_ALPHA)
            + color * OVERLAY_ALPHA
        ).astype(np.uint8)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        thickness = 2 if class_name == "road" else 4
        cv2.drawContours(overlay, contours, -1, OVERLAY_COLORS[class_id], thickness, cv2.LINE_AA)
    return overlay


def save_prediction(
    image_path: Path,
    input_root: Path,
    output_root: Path,
    class_map: np.ndarray,
    image: np.ndarray,
) -> None:
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
        default=DEFAULT_WEIGHTS,
        help="Path to .pt or .onnx weights. With --backend onnx, a .pt suffix is replaced with .onnx.",
    )
    parser.add_argument("--backend", choices=("auto", "pt", "onnx"), default="auto")
    parser.add_argument("--output", type=Path, default=Path("result/lane_detection/yolo_sem_class"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    add_perspective_args(parser)
    add_postprocess_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model = load_semantic_model(args.weights, args.backend)
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
        perspective_config = make_perspective_config(args, image.shape[:2])
        results = model.predict(
            source=image,
            imgsz=args.imgsz,
            device=args.device,
            task="semantic",
            verbose=False,
        )
        class_map = semantic_to_class_map(results[0].semantic_mask, image.shape[:2])
        if args.postprocess:
            class_map = postprocess_class_map(class_map)
        image = apply_perspective(image, perspective_config)
        class_map = apply_perspective(class_map, perspective_config, cv2.INTER_NEAREST)
        save_prediction(source, args.input, args.output, class_map, image)
        print(f"[{index:>3}/{len(sources)}] {source}")

    print(f"Saved classified lane semantic results to {args.output}")


if __name__ == "__main__":
    main()
