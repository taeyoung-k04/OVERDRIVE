#!/usr/bin/env python3
"""Extract lane markings after selecting only the camera's own road."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def _component_touching_ego(candidate: np.ndarray) -> np.ndarray:
    """Select exactly one asphalt component anchored below the camera.

    Pixels nearer the true camera centre and image bottom receive a larger vote,
    and only the best connected component is returned. This keeps neighbouring
    roads from being retained when they enter the lower seed area at bends.
    """
    height, width = candidate.shape
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, 8)
    if count <= 1:
        return np.zeros_like(candidate)

    y0 = int(height * 0.88)
    x0, x1 = int(width * 0.34), int(width * 0.66)
    yy, xx = np.mgrid[y0:height, x0:x1]
    centre_distance = np.abs(xx - (width - 1) / 2) / max(1.0, width * 0.16)
    bottom_distance = (height - 1 - yy) / max(1.0, height * 0.12)
    weights = np.exp(-2.2 * centre_distance**2) * (1.0 + 1.5 * (1.0 - bottom_distance))

    seed_labels = labels[y0:height, x0:x1]
    minimum_area = width * height * 0.008
    best_id = 0
    best_score = -1.0
    for idx in range(1, count):
        if stats[idx, cv2.CC_STAT_AREA] < minimum_area:
            continue
        score = float(weights[seed_labels == idx].sum())
        if score > best_score:
            best_id, best_score = idx, score

    # A fragmented image can leave no large component. In that case, still use
    # the component with the strongest bottom-centre support rather than a side.
    if best_id == 0:
        for idx in range(1, count):
            score = float(weights[seed_labels == idx].sum())
            if score > best_score:
                best_id, best_score = idx, score

    if best_id == 0 or best_score <= 0:
        return np.zeros_like(candidate)
    return ((labels == best_id).astype(np.uint8) * 255)


def _fill_enclosed_holes(mask: np.ndarray) -> np.ndarray:
    """Fill holes inside a road without expanding across a median."""
    inverse = cv2.bitwise_not(mask)
    _, labels, _, _ = cv2.connectedComponentsWithStats(inverse, 8)
    border_ids = np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1])))
    holes = (inverse > 0) & ~np.isin(labels, border_ids)
    result = mask.copy()
    result[holes] = 255
    return result


def _trace_ego_corridor(mask: np.ndarray) -> np.ndarray:
    """Follow the bottom-centre road upward and prune branches after a split."""
    height, width = mask.shape
    traced = np.zeros_like(mask)
    previous: tuple[int, int] | None = None
    max_shift = max(8, int(width * 0.018))

    for y in range(height - 1, int(height * 0.28), -1):
        row = mask[y] > 0
        changes = np.diff(np.pad(row.astype(np.int8), (1, 1)))
        starts = np.flatnonzero(changes == 1)
        ends = np.flatnonzero(changes == -1) - 1
        runs = list(zip(starts.tolist(), ends.tolist()))
        if not runs:
            continue

        if previous is None:
            centre = (width - 1) / 2
            containing = [run for run in runs if run[0] <= centre <= run[1]]
            if containing:
                chosen = max(containing, key=lambda run: run[1] - run[0])
            else:
                chosen = min(runs, key=lambda run: min(abs(run[0] - centre), abs(run[1] - centre)))
        else:
            left, right = previous
            plausible = [
                run for run in runs
                if min(run[1], right + max_shift) >= max(run[0], left - max_shift)
            ]
            if not plausible:
                continue
            previous_centre = (left + right) / 2
            chosen = max(
                plausible,
                key=lambda run: (
                    max(0, min(run[1], right) - max(run[0], left) + 1),
                    -abs((run[0] + run[1]) / 2 - previous_centre),
                ),
            )

        traced[y, chosen[0] : chosen[1] + 1] = 255
        previous = chosen

    vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 9))
    return cv2.morphologyEx(traced, cv2.MORPH_CLOSE, vertical)


def find_ego_road(image: np.ndarray) -> np.ndarray:
    """Segment the asphalt component directly connected to the camera."""
    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.int16)

    # Estimate asphalt colour from several patches at the bottom. The median is
    # insensitive to a white dash passing through one of the patches.
    y0, y1 = int(height * 0.82), int(height * 0.98)
    x0, x1 = int(width * 0.18), int(width * 0.82)
    samples = lab[y0:y1:4, x0:x1:4].reshape(-1, 3)
    sample_value = hsv[y0:y1:4, x0:x1:4, 2].reshape(-1)
    samples = samples[sample_value < np.percentile(sample_value, 72)]
    asphalt = np.median(samples, axis=0)
    distance = np.sqrt(
        0.60 * (lab[..., 0] - asphalt[0]) ** 2
        + 1.25 * (lab[..., 1] - asphalt[1]) ** 2
        + 1.25 * (lab[..., 2] - asphalt[2]) ** 2
    )

    candidate = (
        (distance < 43)
        & (hsv[..., 2] < 185)
        & (hsv[..., 1] < 125)
        & (np.indices((height, width))[0] > int(height * 0.28))
    ).astype(np.uint8) * 255

    scale = max(3, int(round(width / 480)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * scale + 1, 2 * scale + 1))
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel, iterations=2)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, kernel, iterations=1)
    road = _component_touching_ego(candidate)

    # Fill only enclosed gaps. A large closing kernel used here previously could
    # cross the green separator and reconnect the neighbouring road.
    road = _fill_enclosed_holes(road)
    return _trace_ego_corridor(road)


def extract_lane_mask(image: np.ndarray, road: np.ndarray) -> np.ndarray:
    """Extract white paint in or immediately bordering the ego-road mask."""
    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    b, g, r = cv2.split(image)

    # White paint is bright and nearly neutral. The RGB floor prevents bright
    # green fabric from passing only because its HSV saturation is locally low.
    white = (
        (hsv[..., 2] >= 155)
        & (hsv[..., 1] <= 78)
        & (b >= 135)
        & (g >= 135)
        & (r >= 135)
    ).astype(np.uint8) * 255

    # Include solid edge lines whose centre may sit just outside the asphalt
    # component, but do not dilate far enough to reach a separated road.
    # Keep this margin deliberately narrow: in perspective the green median can
    # be only a few dozen pixels wide. A generous dilation would jump across it
    # and admit the edge line of the road behind it.
    margin = max(6, int(round(width * 0.0055)))
    road_corridor = cv2.dilate(
        road,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1)),
    )
    lane = cv2.bitwise_and(white, road_corridor)

    # All legitimate markings occur on the floor. This also guards against a
    # rare road-mask leak into bright wall signs near the horizon.
    lane[: int(height * 0.40)] = 0
    clean = max(2, int(round(width / 960)))
    lane = cv2.morphologyEx(
        lane,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (clean, clean)),
    )

    # Discard tiny reflections/noise, retaining both dashed centre markings and
    # long curved boundary markings.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(lane, 8)
    result = np.zeros_like(lane)
    min_area = max(30, int(width * height * 0.000018))
    for idx in range(1, count):
        x, y, component_width, component_height, area = stats[idx]
        # At long range, painted lane pieces lie roughly horizontally in the
        # image. This rejects vertical furniture/sign fragments accidentally
        # touching an asphalt-coloured shadow without assuming a route side.
        far_vertical_object = (
            y + component_height / 2 < height * 0.52
            and component_width < component_height * 1.35
        )
        if (
            area >= min_area
            and max(component_width, component_height) >= max(7, width // 180)
            and not far_vertical_object
        ):
            result[labels == idx] = 255
    return result


def make_overlay(image: np.ndarray, lane: np.ndarray, road: np.ndarray) -> np.ndarray:
    overlay = image.copy()
    tint = np.zeros_like(image)
    tint[road > 0] = (35, 20, 0)
    overlay = cv2.addWeighted(overlay, 1.0, tint, 0.24, 0)

    contours, _ = cv2.findContours(lane, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 0, 255), 5, cv2.LINE_AA)
    overlay[lane > 0] = (0, 255, 255)
    return overlay


def process_image(source: Path, input_root: Path, output_root: Path) -> None:
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {source}")

    road = find_ego_road(image)
    lane = extract_lane_mask(image, road)
    overlay = make_overlay(image, lane, road)
    relative = source.relative_to(input_root)

    destinations = {
        "overlay": overlay,
        "lane_mask": lane,
        "road_mask": road,
    }
    for directory, result in destinations.items():
        destination = output_root / directory / relative.with_suffix(".png")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(destination), result):
            raise RuntimeError(f"Could not write image: {destination}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("dataset/lane_detection/frames"))
    parser.add_argument("--output", type=Path, default=Path("result/lane_detection/rules"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sources = sorted(
        path for path in args.input.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not sources:
        raise SystemExit(f"No images found below {args.input}")

    for index, source in enumerate(sources, 1):
        process_image(source, args.input, args.output)
        print(f"[{index:>3}/{len(sources)}] {source}")
    print(f"Saved masks and overlays to {args.output}")


if __name__ == "__main__":
    main()
