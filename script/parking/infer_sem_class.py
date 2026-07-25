#!/usr/bin/env python3
"""Run parking semantic segmentation and save class maps and overlays."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


CLASS_NAMES = ("background", "road", "park", "grass", "lane")
CLASS_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}
COLORS = {
    CLASS_TO_ID["road"]: (70, 70, 70),
    CLASS_TO_ID["park"]: (255, 160, 40),
    CLASS_TO_ID["grass"]: (40, 180, 40),
    CLASS_TO_ID["lane"]: (0, 230, 255),
}
DEFAULT_WEIGHTS = Path(
    "runs/semantic/yolo_parking_sem_class/"
    "train_cpu_640_yolo26n_5class/weights/best.pt"
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def resolve_model_path(weights: Path, backend: str) -> Path:
    if backend == "auto":
        path = weights
    else:
        suffix = f".{backend}"
        path = weights if weights.suffix.lower() == suffix else weights.with_suffix(suffix)
    if not path.exists():
        raise SystemExit(f"Model file does not exist: {path}")
    return path


def semantic_to_class_map(semantic: object, shape: tuple[int, int]) -> np.ndarray:
    if semantic is None:
        return np.zeros(shape, dtype=np.uint8)
    class_map = semantic.data.cpu().numpy().astype(np.uint8)
    if class_map.shape != shape:
        class_map = cv2.resize(
            class_map, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST
        )
    return class_map


def make_overlay(image: np.ndarray, class_map: np.ndarray) -> np.ndarray:
    output = image.copy()
    alpha = 0.35
    for class_id, color in COLORS.items():
        selected = class_map == class_id
        if not np.any(selected):
            continue
        output[selected] = (
            image[selected].astype(np.float32) * (1.0 - alpha)
            + np.asarray(color, dtype=np.float32) * alpha
        ).astype(np.uint8)
    return output


def save_results(
    source: Path,
    input_root: Path,
    output_root: Path,
    image: np.ndarray,
    class_map: np.ndarray,
) -> None:
    relative = source.relative_to(input_root).with_suffix(".png")
    results = {"overlay": make_overlay(image, class_map), "class_map": class_map}
    for name in CLASS_NAMES[1:]:
        results[f"{name}_mask"] = (
            (class_map == CLASS_TO_ID[name]).astype(np.uint8) * 255
        )
    for directory, result in results.items():
        destination = output_root / directory / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(destination), result):
            raise RuntimeError(f"Could not write image: {destination}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("dataset/parking/perspective"))
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--backend", choices=("auto", "pt", "onnx"), default="auto")
    parser.add_argument("--output", type=Path, default=Path("result/parking/yolo_sem_class"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "ultralytics is not installed in the active environment."
        ) from exc

    model_path = resolve_model_path(args.weights, args.backend)
    model = YOLO(str(model_path), task="semantic")
    sources = sorted(
        path
        for path in args.input.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not sources:
        raise SystemExit(f"No images found below {args.input}")

    for source in tqdm(sources, desc="parking", unit="image"):
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read image: {source}")
        predictions = model.predict(
            source=image,
            imgsz=args.imgsz,
            device=args.device,
            task="semantic",
            verbose=False,
        )
        class_map = semantic_to_class_map(
            predictions[0].semantic_mask, image.shape[:2]
        )
        save_results(source, args.input, args.output, image, class_map)
    tqdm.write(f"Saved parking semantic results to {args.output}")


if __name__ == "__main__":
    main()
