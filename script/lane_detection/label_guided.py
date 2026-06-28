#!/usr/bin/env python3
"""Extract lane markings using hand-labeled road/lane masks as guidance."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from rules import IMAGE_SUFFIXES, find_ego_road, make_overlay


FRAME_RE = re.compile(r"frame_(\d+)s")


@dataclass(frozen=True)
class LaneThresholds:
    v_min: int
    s_max: int
    b_min: int
    g_min: int
    r_min: int
    l_min: int


def frame_second(path: Path) -> int:
    match = FRAME_RE.search(path.stem)
    if not match:
        raise ValueError(f"Could not parse frame second from {path}")
    return int(match.group(1))


def read_mask(path: Path, shape: tuple[int, int] | None = None) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise RuntimeError(f"Could not read mask: {path}")

    if mask.ndim == 3 and mask.shape[2] == 4:
        binary = mask[..., 3] > 8
    elif mask.ndim == 3:
        binary = np.any(mask[..., :3] > 8, axis=2)
    else:
        binary = mask > 8

    result = binary.astype(np.uint8) * 255
    if shape is not None and result.shape != shape:
        result = cv2.resize(result, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return result


class LabelStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self._masks: dict[tuple[str, str], dict[int, Path]] = {}
        for path in root.rglob("*.png"):
            if path.name.endswith("~"):
                continue
            try:
                route, kind = path.relative_to(root).parts[:2]
            except ValueError:
                continue
            if kind not in {"road", "lane"}:
                continue
            self._masks.setdefault((route, kind), {})[frame_second(path)] = path

    def nearest(
        self,
        route: str,
        kind: str,
        second: int,
        max_delta: int,
        shape: tuple[int, int],
    ) -> np.ndarray | None:
        masks = self._masks.get((route, kind), {})
        if not masks:
            return None
        nearest_second = min(masks, key=lambda value: abs(value - second))
        if abs(nearest_second - second) > max_delta:
            return None
        return read_mask(masks[nearest_second], shape)

    def exact(self, route: str, kind: str, second: int, shape: tuple[int, int]) -> np.ndarray | None:
        path = self._masks.get((route, kind), {}).get(second)
        return read_mask(path, shape) if path else None

    def labeled_lane_pairs(self, image_root: Path) -> list[tuple[Path, Path]]:
        pairs: list[tuple[Path, Path]] = []
        for (route, kind), masks in self._masks.items():
            if kind != "lane":
                continue
            for second, mask_path in sorted(masks.items()):
                image_path = image_root / route / f"frame_{second:06d}s.jpg"
                if image_path.exists():
                    pairs.append((image_path, mask_path))
        return pairs


def build_lane_thresholds(image_root: Path, labels: LabelStore) -> LaneThresholds:
    hsv_values: list[np.ndarray] = []
    bgr_values: list[np.ndarray] = []
    lab_values: list[np.ndarray] = []

    for image_path, lane_path in labels.labeled_lane_pairs(image_root):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        lane = read_mask(lane_path, image.shape[:2]) > 0
        if not np.any(lane):
            continue

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        pixels = np.flatnonzero(lane)
        if pixels.size > 30000:
            pixels = pixels[:: max(1, pixels.size // 30000)]
        hsv_values.append(hsv.reshape(-1, 3)[pixels])
        bgr_values.append(image.reshape(-1, 3)[pixels])
        lab_values.append(lab.reshape(-1, 3)[pixels])

    if not hsv_values:
        raise SystemExit("No usable lane labels found. Check --label.")

    hsv = np.concatenate(hsv_values, axis=0)
    bgr = np.concatenate(bgr_values, axis=0)
    lab = np.concatenate(lab_values, axis=0)

    return LaneThresholds(
        v_min=int(max(120, np.percentile(hsv[:, 2], 3) - 12)),
        s_max=int(min(120, np.percentile(hsv[:, 1], 98) + 12)),
        b_min=int(max(100, np.percentile(bgr[:, 0], 3) - 18)),
        g_min=int(max(100, np.percentile(bgr[:, 1], 3) - 18)),
        r_min=int(max(100, np.percentile(bgr[:, 2], 3) - 18)),
        l_min=int(max(120, np.percentile(lab[:, 0], 3) - 12)),
    )


def lane_by_learned_color(image: np.ndarray, thresholds: LaneThresholds) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    b, g, r = cv2.split(image)
    lane = (
        (hsv[..., 2] >= thresholds.v_min)
        & (hsv[..., 1] <= thresholds.s_max)
        & (lab[..., 0] >= thresholds.l_min)
        & (b >= thresholds.b_min)
        & (g >= thresholds.g_min)
        & (r >= thresholds.r_min)
    )
    return lane.astype(np.uint8) * 255


def clean_lane_mask(lane: np.ndarray) -> np.ndarray:
    height, width = lane.shape
    lane[: int(height * 0.36)] = 0

    open_size = max(2, int(round(width / 1100)))
    lane = cv2.morphologyEx(
        lane,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size)),
    )

    count, labels, stats, _ = cv2.connectedComponentsWithStats(lane, 8)
    result = np.zeros_like(lane)
    min_area = max(35, int(width * height * 0.000012))
    for idx in range(1, count):
        x, y, component_width, component_height, area = stats[idx]
        if area < min_area:
            continue
        if max(component_width, component_height) < max(6, width // 220):
            continue
        far_vertical = y + component_height / 2 < height * 0.54 and component_width < component_height * 1.25
        if far_vertical:
            continue
        result[labels == idx] = 255
    return result


def extract_label_guided_lane(
    image: np.ndarray,
    route: str,
    second: int,
    labels: LabelStore,
    thresholds: LaneThresholds,
    max_label_delta: int,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    shape = (height, width)

    road = labels.nearest(route, "road", second, max_label_delta, shape)
    if road is None:
        road = find_ego_road(image)
    else:
        close = max(5, int(round(width * 0.004)))
        road = cv2.morphologyEx(
            road,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close + 1, 2 * close + 1)),
        )

    lane = lane_by_learned_color(image, thresholds)

    road_margin = max(7, int(round(width * 0.006)))
    road_corridor = cv2.dilate(
        road,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * road_margin + 1, 2 * road_margin + 1)),
    )
    lane = cv2.bitwise_and(lane, road_corridor)

    lane_prior = labels.nearest(route, "lane", second, max_label_delta, shape)
    if lane_prior is not None:
        prior_margin = max(18, int(round(width * 0.018)))
        lane_prior = cv2.dilate(
            lane_prior,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * prior_margin + 1, 2 * prior_margin + 1)),
        )
        lane = cv2.bitwise_and(lane, lane_prior)

    return clean_lane_mask(lane), road


def process_image(
    source: Path,
    input_root: Path,
    output_root: Path,
    labels: LabelStore,
    thresholds: LaneThresholds,
    max_label_delta: int,
) -> None:
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {source}")

    route = source.relative_to(input_root).parts[0]
    second = frame_second(source)
    lane, road = extract_label_guided_lane(image, route, second, labels, thresholds, max_label_delta)
    overlay = make_overlay(image, lane, road)

    relative = source.relative_to(input_root)
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
    parser.add_argument("--label", type=Path, default=Path("dataset/lane_detection/labels"))
    parser.add_argument("--output", type=Path, default=Path("result/lane_detection/label_guided"))
    parser.add_argument("--max-label-delta", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels = LabelStore(args.label)
    thresholds = build_lane_thresholds(args.input, labels)
    print(
        "Learned lane thresholds: "
        f"V>={thresholds.v_min}, S<={thresholds.s_max}, "
        f"BGR>=({thresholds.b_min},{thresholds.g_min},{thresholds.r_min}), "
        f"L>={thresholds.l_min}"
    )

    sources = sorted(
        path for path in args.input.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not sources:
        raise SystemExit(f"No images found below {args.input}")

    for index, source in enumerate(sources, 1):
        process_image(source, args.input, args.output, labels, thresholds, args.max_label_delta)
        print(f"[{index:>3}/{len(sources)}] {source}")
    print(f"Saved label-guided masks and overlays to {args.output}")


if __name__ == "__main__":
    main()
