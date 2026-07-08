from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


CLASS_TO_ID = {
    "background": 0,
    "road": 1,
    "lane_left": 2,
    "lane_center": 3,
    "lane_right": 4,
    "stop_line": 5,
}

DEFAULT_POSTPROCESS_CONFIG = Path(".env")
DEFAULT_ROAD_GAP_PX = 48.0
DEFAULT_LANE_GAP_PX = 16.0
DEFAULT_STOP_LINE_MIN_AREA_PX = 1024.0
DEFAULT_POSTPROCESS_REFERENCE_WIDTH = 960.0
DEFAULT_POSTPROCESS_REFERENCE_HEIGHT = 540.0
POSTPROCESS_TOP_REMOVE_RATIO = 0.25


@dataclass(frozen=True)
class PostprocessConfig:
    road_gap_px: float
    lane_gap_px: float
    stop_line_min_area_px: float


def _read_key_value_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        value = value.split("#", 1)[0].strip()
        values[key.strip()] = value.strip("\"'")
    return values


def _get_float_config(values: dict[str, str], key: str, default: float) -> float:
    raw_value = values.get(key, os.getenv(key))
    if raw_value in (None, ""):
        return default

    try:
        return float(raw_value)
    except ValueError as exc:
        raise SystemExit(f"{key} must be a number, got: {raw_value}") from exc


def load_postprocess_config(path: Path) -> PostprocessConfig:
    values = _read_key_value_file(path)
    return PostprocessConfig(
        road_gap_px=_get_float_config(values, "LANE_POSTPROCESS_ROAD_GAP_PX", DEFAULT_ROAD_GAP_PX),
        lane_gap_px=_get_float_config(values, "LANE_POSTPROCESS_LANE_GAP_PX", DEFAULT_LANE_GAP_PX),
        stop_line_min_area_px=_get_float_config(
            values,
            "LANE_POSTPROCESS_STOP_LINE_MIN_AREA_PX",
            DEFAULT_STOP_LINE_MIN_AREA_PX,
        ),
    )


def _distance_to_mask(mask: np.ndarray) -> np.ndarray:
    return cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)


def _main_road_mask(road_mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(road_mask.astype(np.uint8), 8)
    if count <= 1:
        return road_mask.copy()

    bottom_labels = set(int(label) for label in labels[-1, :] if label != 0)
    if not bottom_labels:
        return np.zeros_like(road_mask, dtype=bool)

    main_label = max(bottom_labels, key=lambda label: stats[label, cv2.CC_STAT_AREA])
    return labels == main_label


def _road_row_bounds(road_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height = road_mask.shape[0]
    left_edges = np.full(height, -1, dtype=np.int32)
    right_edges = np.full(height, -1, dtype=np.int32)

    for y in range(height):
        xs = np.flatnonzero(road_mask[y])
        if xs.size == 0:
            continue
        left_edges[y] = int(xs[0])
        right_edges[y] = int(xs[-1])
    return left_edges, right_edges


def _road_side_edge_masks(road_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left_edges, right_edges = _road_row_bounds(road_mask)
    left_edge_mask = np.zeros_like(road_mask, dtype=bool)
    right_edge_mask = np.zeros_like(road_mask, dtype=bool)

    ys = np.flatnonzero(left_edges >= 0)
    if ys.size == 0:
        return left_edge_mask, right_edge_mask

    left_edge_mask[ys, left_edges[ys]] = True
    right_edge_mask[ys, right_edges[ys]] = True
    return left_edge_mask, right_edge_mask


def _distance_threshold_px(shape: tuple[int, int], value_px: float) -> float:
    height, width = shape[:2]
    width_scale = float(width) / DEFAULT_POSTPROCESS_REFERENCE_WIDTH
    height_scale = float(height) / DEFAULT_POSTPROCESS_REFERENCE_HEIGHT
    return value_px * ((width_scale + height_scale) * 0.5)


def _area_threshold_px(shape: tuple[int, int], value_px: float) -> float:
    height, width = shape[:2]
    area_scale = float(width * height) / (
        DEFAULT_POSTPROCESS_REFERENCE_WIDTH * DEFAULT_POSTPROCESS_REFERENCE_HEIGHT
    )
    return value_px * area_scale


def _remove_small_components(mask: np.ndarray, min_area_px: float) -> np.ndarray:
    remove = np.zeros_like(mask, dtype=bool)
    if min_area_px <= 0 or not np.any(mask):
        return remove

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] < min_area_px:
            remove[labels == label] = True
    return remove


def postprocess_class_map(class_map: np.ndarray, config: PostprocessConfig) -> np.ndarray:
    processed = class_map.copy()
    top_cutoff = int(round(processed.shape[0] * POSTPROCESS_TOP_REMOVE_RATIO))
    if top_cutoff > 0:
        processed[:top_cutoff, :] = CLASS_TO_ID["background"]

    road_gap_px = _distance_threshold_px(class_map.shape, config.road_gap_px)
    lane_gap_px = _distance_threshold_px(class_map.shape, config.lane_gap_px)
    stop_line_min_area_px = _area_threshold_px(class_map.shape, config.stop_line_min_area_px)
    road_id = CLASS_TO_ID["road"]
    road_mask = processed == road_id
    main_road = _main_road_mask(road_mask)
    if not np.any(main_road):
        processed[road_mask] = CLASS_TO_ID["background"]
        for class_name in ("lane_left", "lane_center", "lane_right", "stop_line"):
            processed[processed == CLASS_TO_ID[class_name]] = CLASS_TO_ID["background"]
        return processed

    count, labels, _, _ = cv2.connectedComponentsWithStats(road_mask.astype(np.uint8), 8)
    main_distance = _distance_to_mask(main_road)
    merged_main_road = main_road.copy()
    for label in range(1, count):
        component = labels == label
        if np.any(component & main_road):
            continue
        if float(main_distance[component].min()) < road_gap_px:
            merged_main_road |= component

    processed[road_mask & ~merged_main_road] = CLASS_TO_ID["background"]
    processed[merged_main_road] = road_id

    distance_to_road = _distance_to_mask(merged_main_road)
    left_edge_mask, right_edge_mask = _road_side_edge_masks(merged_main_road)
    distance_to_left_edge = _distance_to_mask(left_edge_mask)
    distance_to_right_edge = _distance_to_mask(right_edge_mask)

    lane_left_id = CLASS_TO_ID["lane_left"]
    lane_left_mask = processed == lane_left_id
    processed[lane_left_mask & (distance_to_left_edge >= lane_gap_px)] = CLASS_TO_ID["background"]

    lane_right_id = CLASS_TO_ID["lane_right"]
    lane_right_mask = processed == lane_right_id
    processed[lane_right_mask & (distance_to_right_edge >= lane_gap_px)] = CLASS_TO_ID["background"]

    lane_center_id = CLASS_TO_ID["lane_center"]
    lane_center_mask = processed == lane_center_id
    processed[lane_center_mask & (distance_to_road >= lane_gap_px)] = CLASS_TO_ID["background"]

    stop_line_id = CLASS_TO_ID["stop_line"]
    stop_line_mask = processed == stop_line_id
    processed[stop_line_mask & (distance_to_road >= lane_gap_px)] = CLASS_TO_ID["background"]
    stop_line_after_distance = processed == stop_line_id
    processed[_remove_small_components(stop_line_after_distance, stop_line_min_area_px)] = CLASS_TO_ID[
        "background"
    ]
    return processed


def add_postprocess_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--postprocess", action="store_true", help="Apply road/lane semantic postprocessing.")
    parser.add_argument(
        "--postprocess-config",
        type=Path,
        default=DEFAULT_POSTPROCESS_CONFIG,
        help="Path to an env-style postprocess config file.",
    )

