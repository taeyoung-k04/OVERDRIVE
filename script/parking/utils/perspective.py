from __future__ import annotations

import argparse
from dataclasses import dataclass

import cv2
import numpy as np


DEFAULT_BEV_SRC_POINTS = np.array(
    [
        [0.4052, 0.4731],
        [0.6391, 0.4731],
        [0.7896, 0.9046],
        [0.2427, 0.9046],
    ],
    dtype=np.float32,
)
DEFAULT_BEV_OUTPUT_WIDTH_RATIO = 0.50
DEFAULT_BEV_OUTPUT_HEIGHT_TO_WIDTH = 0.75
DEFAULT_BEV_DST_LEFT_MARGIN_RATIO = 0.35
DEFAULT_BEV_DST_RIGHT_MARGIN_RATIO = 0.35
DEFAULT_BEV_DST_TOP_MARGIN_RATIO = 0.10
DEFAULT_BEV_DST_BOTTOM_MARGIN_RATIO = 0.06


@dataclass(frozen=True)
class PerspectiveConfig:
    src_points: np.ndarray
    dst_points: np.ndarray
    output_size: tuple[int, int]


def _scale_normalized_points(points: np.ndarray, width: int, height: int) -> np.ndarray:
    scaled = points.copy()
    scaled[:, 0] *= width - 1
    scaled[:, 1] *= height - 1
    return scaled.astype(np.float32)


def _make_bev_destination(width: int) -> tuple[np.ndarray, tuple[int, int]]:
    output_width = max(1, int(round(width * DEFAULT_BEV_OUTPUT_WIDTH_RATIO)))
    output_height = max(1, int(round(output_width * DEFAULT_BEV_OUTPUT_HEIGHT_TO_WIDTH)))
    left = output_width * DEFAULT_BEV_DST_LEFT_MARGIN_RATIO
    right = output_width * (1.0 - DEFAULT_BEV_DST_RIGHT_MARGIN_RATIO)
    top = output_height * DEFAULT_BEV_DST_TOP_MARGIN_RATIO
    bottom = output_height * (1.0 - DEFAULT_BEV_DST_BOTTOM_MARGIN_RATIO)
    dst_points = np.array(
        [
            [left, top],
            [right, top],
            [right, bottom],
            [left, bottom],
        ],
        dtype=np.float32,
    )
    return dst_points, (output_width, output_height)


def add_perspective_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--perspective",
        action="store_true",
        help="Apply bird's-eye-view perspective warp after postprocessing.",
    )


def make_perspective_config(args: argparse.Namespace, shape: tuple[int, int]) -> PerspectiveConfig | None:
    if not getattr(args, "perspective", False):
        return None

    height, width = shape[:2]
    src_points = _scale_normalized_points(DEFAULT_BEV_SRC_POINTS, width, height)
    dst_points, output_size = _make_bev_destination(width)
    return PerspectiveConfig(src_points=src_points, dst_points=dst_points, output_size=output_size)


def apply_perspective(
    image: np.ndarray,
    config: PerspectiveConfig | None,
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    if config is None:
        return image

    transform = cv2.getPerspectiveTransform(config.src_points, config.dst_points)
    return cv2.warpPerspective(image, transform, config.output_size, flags=interpolation)
