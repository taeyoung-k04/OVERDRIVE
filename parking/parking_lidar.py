#!/usr/bin/env python3
"""
LiDAR-only autonomous parking controller.

Hardware architecture
---------------------
* 2-D USB LiDAR -> this Python process
* Arduino Uno/Nano -> this Python process over USB serial
* Arduino firmware is NOT modified. This program only sends:
      C,<steering -1000..1000>,<drive -255..255>\n
      X\n

Default LiDAR adapter
---------------------
The built-in adapter targets SLAMTEC RPLIDAR devices through the Python
``rplidar`` package. For another USB LiDAR, implement ``LidarSource`` and keep
all perception/planning/control code unchanged.

Important
---------
This is a configurable engineering starting point, not a zero-tuning drop-in.
Measure the vehicle dimensions, LiDAR pose, parking geometry, and low-speed
motion parameters before running with the wheels on the floor.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import queue
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Iterable, Iterator, Optional, Protocol, Sequence

import numpy as np
import serial
from scipy.spatial import cKDTree


# -----------------------------------------------------------------------------
# Math helpers
# -----------------------------------------------------------------------------


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def angle_diff(target: float, current: float) -> float:
    return wrap_angle(target - current)


def rotation(theta: float) -> np.ndarray:
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float

    def matrix(self) -> np.ndarray:
        c = math.cos(self.yaw)
        s = math.sin(self.yaw)
        return np.array(
            [[c, -s, self.x], [s, c, self.y], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    @staticmethod
    def from_matrix(matrix: np.ndarray) -> "Pose2D":
        return Pose2D(
            x=float(matrix[0, 2]),
            y=float(matrix[1, 2]),
            yaw=math.atan2(float(matrix[1, 0]), float(matrix[0, 0])),
        )

    def compose(self, other: "Pose2D") -> "Pose2D":
        return Pose2D.from_matrix(self.matrix() @ other.matrix())

    def inverse(self) -> "Pose2D":
        return Pose2D.from_matrix(np.linalg.inv(self.matrix()))

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if len(points) == 0:
            return points.copy()
        return points @ rotation(self.yaw).T + np.array([self.x, self.y])


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


@dataclass
class Config:
    # Serial
    arduino_port: str = "COM8"
    arduino_baud: int = 115200
    lidar_port: str = "COM5"
    lidar_baud: int = 115200

    # LiDAR installation. Coordinates are in the vehicle frame:
    # +x = vehicle forward, +y = vehicle left.
    lidar_x_m: float = -0.42
    lidar_y_m: float = 0.0
    lidar_yaw_deg: float = 180.0
    lidar_clockwise_angles: bool = False
    lidar_angle_offset_deg: float = 0.0

    # Scan filtering
    min_range_m: float = 0.12
    max_range_m: float = 8.0
    voxel_size_m: float = 0.025
    scan_min_points: int = 140
    max_scan_age_s: float = 0.35

    # Clustering / parked-car geometry
    cluster_jump_base_m: float = 0.07
    cluster_jump_range_gain: float = 0.015
    cluster_min_points: int = 7
    cluster_min_length_m: float = 0.25
    cluster_max_length_m: float = 2.20
    car_pair_parallel_deg: float = 20.0
    expected_slot_width_m: float = 0.95
    slot_width_tolerance_m: float = 0.40
    slot_depth_m: float = 1.55
    goal_depth_from_car_centers_m: float = 0.0
    detection_rear_only: bool = True
    detection_x_min_m: float = -5.0
    detection_x_max_m: float = 1.0
    detection_abs_y_max_m: float = 4.0

    # Vehicle footprint and Ackermann model
    wheelbase_m: float = 0.72
    vehicle_length_m: float = 1.10
    vehicle_width_m: float = 0.63
    rear_axle_to_rear_m: float = 0.23
    safety_margin_m: float = 0.055
    max_steering_deg: float = 29.0

    # Motion model used by Hybrid A* and receding-horizon control
    forward_speed_mps: float = 0.28
    reverse_speed_mps: float = 0.22
    forward_pwm: int = 185
    reverse_pwm: int = -185
    planner_step_m: float = 0.12
    planner_xy_resolution_m: float = 0.08
    planner_yaw_resolution_deg: float = 10.0
    planner_max_nodes: int = 70000
    planner_bounds_margin_m: float = 1.6
    planner_collision_sample_m: float = 0.075
    planner_goal_xy_tolerance_m: float = 0.13
    planner_goal_yaw_tolerance_deg: float = 12.0
    planner_reverse_cost: float = 1.05
    planner_gear_change_cost: float = 0.85
    planner_steering_cost: float = 0.12
    planner_steering_change_cost: float = 0.22

    # ICP LiDAR odometry
    icp_max_iterations: int = 18
    icp_max_correspondence_m: float = 0.25
    icp_trim_fraction: float = 0.78
    icp_min_pairs: int = 45
    icp_max_rmse_m: float = 0.09
    icp_max_translation_per_scan_m: float = 0.18
    icp_max_rotation_per_scan_deg: float = 12.0

    # Online controller. Commands are deliberately short; pose is re-measured
    # from the next LiDAR scans instead of trusting wheel odometry.
    control_hz: float = 18.0
    command_horizon_s: float = 0.28
    command_steering_levels: tuple[int, ...] = (-1000, -600, 0, 600, 1000)
    tracking_lookahead_m: float = 0.28
    stop_clearance_m: float = 0.17
    goal_hold_s: float = 3.5
    goal_stable_s: float = 0.45
    max_path_error_m: float = 0.42
    replan_interval_s: float = 0.8

    # Exit target expressed in the detected slot coordinate frame.
    # slot +x follows the parked cars' longitudinal direction toward the aisle;
    # slot +y points from the left obstacle toward the right obstacle.
    exit_forward_m: float = 2.0
    exit_lateral_m: float = 0.0
    exit_yaw_offset_deg: float = 0.0

    # Runtime
    startup_scans: int = 6
    dry_run: bool = False
    live_plot: bool = False
    save_debug_npz: str = "parking_debug.npz"

    @staticmethod
    def load(path: Path) -> "Config":
        cfg = Config()
        if not path.exists():
            return cfg
        data = json.loads(path.read_text(encoding="utf-8"))
        for key, value in data.items():
            if not hasattr(cfg, key):
                raise ValueError(f"Unknown config key: {key}")
            if key == "command_steering_levels":
                value = tuple(int(v) for v in value)
            setattr(cfg, key, value)
        return cfg


# -----------------------------------------------------------------------------
# Hardware I/O
# -----------------------------------------------------------------------------


class LidarSource(Protocol):
    def start(self) -> None: ...

    def get_latest(self, timeout: float) -> tuple[float, np.ndarray]: ...

    def close(self) -> None: ...


class RPLidarSource:
    """Latest-scan adapter for the common ``rplidar`` Python package."""

    def __init__(self, port: str, baudrate: int) -> None:
        self.port = port
        self.baudrate = baudrate
        self._lidar = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._latest: "queue.Queue[tuple[float, np.ndarray]]" = queue.Queue(maxsize=1)
        self._error: Optional[BaseException] = None

    def start(self) -> None:
        try:
            from rplidar import RPLidar  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Install the RPLIDAR driver first: pip install rplidar"
            ) from exc

        self._lidar = RPLidar(self.port, baudrate=self.baudrate, timeout=2)
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="rplidar", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        assert self._lidar is not None
        try:
            # Each measurement is normally (quality, angle_deg, distance_mm).
            for scan in self._lidar.iter_scans():
                if self._stop.is_set():
                    break
                rows = np.asarray(
                    [(float(angle), float(distance) / 1000.0) for _, angle, distance in scan],
                    dtype=np.float64,
                )
                if rows.size == 0:
                    continue
                item = (time.monotonic(), rows)
                try:
                    self._latest.get_nowait()
                except queue.Empty:
                    pass
                self._latest.put_nowait(item)
        except BaseException as exc:  # propagated on get_latest
            self._error = exc

    def get_latest(self, timeout: float) -> tuple[float, np.ndarray]:
        if self._error is not None:
            raise RuntimeError(f"LiDAR reader stopped: {self._error}") from self._error
        try:
            return self._latest.get(timeout=timeout)
        except queue.Empty as exc:
            if self._error is not None:
                raise RuntimeError(f"LiDAR reader stopped: {self._error}") from self._error
            raise TimeoutError("Timed out waiting for a LiDAR scan") from exc

    def close(self) -> None:
        self._stop.set()
        lidar = self._lidar
        if lidar is not None:
            for method_name in ("stop", "stop_motor", "disconnect"):
                try:
                    method = getattr(lidar, method_name, None)
                    if method is not None:
                        method()
                except Exception:
                    pass
        if self._thread is not None:
            self._thread.join(timeout=1.5)


class ArduinoLink:
    """Keeps the existing Arduino serial protocol unchanged."""

    def __init__(self, port: str, baudrate: int, dry_run: bool = False) -> None:
        self.port = port
        self.baudrate = baudrate
        self.dry_run = dry_run
        self.serial: Optional[serial.Serial] = None
        self._reader: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.last_lines: "queue.Queue[str]" = queue.Queue(maxsize=100)
        self._lock = threading.Lock()

    def open(self) -> None:
        if self.dry_run:
            print("[DRY RUN] Arduino serial output disabled")
            return
        self.serial = serial.Serial(
            self.port,
            self.baudrate,
            timeout=0.05,
            write_timeout=0.15,
        )
        time.sleep(2.1)  # Uno/Nano resets when the serial port opens.
        self._stop.clear()
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    def _read_loop(self) -> None:
        assert self.serial is not None
        while not self._stop.is_set():
            try:
                raw = self.serial.readline()
                if not raw:
                    continue
                line = raw.decode("utf-8", errors="replace").strip()
                if line:
                    try:
                        self.last_lines.put_nowait(line)
                    except queue.Full:
                        try:
                            self.last_lines.get_nowait()
                            self.last_lines.put_nowait(line)
                        except queue.Empty:
                            pass
                    if line.startswith(("FAULT", "WATCHDOG", "ERR")):
                        print(f"[Arduino] {line}", file=sys.stderr)
            except Exception as exc:
                if not self._stop.is_set():
                    print(f"Arduino reader error: {exc}", file=sys.stderr)
                return

    def send_control(self, steering: int, drive_pwm: int) -> None:
        steering = int(np.clip(steering, -1000, 1000))
        drive_pwm = int(np.clip(drive_pwm, -255, 255))
        line = f"C,{steering},{drive_pwm}\n"
        if self.dry_run:
            return
        if self.serial is None:
            raise RuntimeError("Arduino serial port is not open")
        with self._lock:
            self.serial.write(line.encode("ascii"))

    def stop_vehicle(self) -> None:
        if self.dry_run:
            return
        if self.serial is None:
            return
        try:
            with self._lock:
                self.serial.write(b"X\n")
                self.serial.flush()
        except Exception:
            pass

    def close(self) -> None:
        self.stop_vehicle()
        self._stop.set()
        if self._reader is not None:
            self._reader.join(timeout=0.6)
        if self.serial is not None:
            try:
                self.serial.close()
            except Exception:
                pass


# -----------------------------------------------------------------------------
# Scan conversion and geometry
# -----------------------------------------------------------------------------


def polar_scan_to_vehicle_points(scan: np.ndarray, cfg: Config) -> np.ndarray:
    """Convert [angle_deg, range_m] rows to the vehicle coordinate frame."""
    if scan.ndim != 2 or scan.shape[1] != 2:
        raise ValueError("LiDAR scan must have shape (N, 2): angle_deg, distance_m")

    angle_deg = scan[:, 0].astype(np.float64)
    distance = scan[:, 1].astype(np.float64)
    valid = (
        np.isfinite(angle_deg)
        & np.isfinite(distance)
        & (distance >= cfg.min_range_m)
        & (distance <= cfg.max_range_m)
    )
    angle_deg = angle_deg[valid]
    distance = distance[valid]

    direction = -1.0 if cfg.lidar_clockwise_angles else 1.0
    theta = np.deg2rad(direction * angle_deg + cfg.lidar_angle_offset_deg)
    lidar_points = np.column_stack((distance * np.cos(theta), distance * np.sin(theta)))

    lidar_pose = Pose2D(
        cfg.lidar_x_m,
        cfg.lidar_y_m,
        math.radians(cfg.lidar_yaw_deg),
    )
    points = lidar_pose.transform_points(lidar_points)

    # Remove returns from the host vehicle itself. A rear-mounted 360-degree
    # LiDAR often sees the body, seat, bracket, or bumper. Keeping those points
    # would make the planner believe the start pose is already in collision.
    front = cfg.vehicle_length_m - cfg.rear_axle_to_rear_m
    rear = cfg.rear_axle_to_rear_m
    half_width = cfg.vehicle_width_m / 2.0
    self_mask = (
        (points[:, 0] >= -rear - 0.04)
        & (points[:, 0] <= front + 0.04)
        & (np.abs(points[:, 1]) <= half_width + 0.04)
    )
    points = points[~self_mask]
    return voxel_downsample(points, cfg.voxel_size_m)


def voxel_downsample(points: np.ndarray, voxel: float) -> np.ndarray:
    if len(points) == 0 or voxel <= 0:
        return points.copy()
    keys = np.floor(points / voxel).astype(np.int64)
    _, indices = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(indices)]


def cluster_ordered_scan(points: np.ndarray, cfg: Config) -> list[np.ndarray]:
    """
    Cluster an angle-ordered scan with a range-aware Euclidean jump threshold.
    The RPLIDAR scan is angle ordered; rotation/translation preserves order.
    """
    if len(points) == 0:
        return []

    ranges = np.linalg.norm(points, axis=1)
    clusters: list[list[np.ndarray]] = [[points[0]]]
    for index in range(1, len(points)):
        gap = float(np.linalg.norm(points[index] - points[index - 1]))
        threshold = cfg.cluster_jump_base_m + cfg.cluster_jump_range_gain * min(
            ranges[index], ranges[index - 1]
        )
        if gap > threshold:
            clusters.append([])
        clusters[-1].append(points[index])

    # Connect first and last cluster for a 360-degree scan when appropriate.
    if len(clusters) > 1:
        first = np.asarray(clusters[0])
        last = np.asarray(clusters[-1])
        wrap_gap = float(np.linalg.norm(first[0] - last[-1]))
        threshold = cfg.cluster_jump_base_m + cfg.cluster_jump_range_gain * min(
            np.linalg.norm(first[0]), np.linalg.norm(last[-1])
        )
        if wrap_gap <= threshold:
            clusters[0] = clusters[-1] + clusters[0]
            clusters.pop()

    output: list[np.ndarray] = []
    for raw in clusters:
        cluster = np.asarray(raw, dtype=np.float64)
        if len(cluster) < cfg.cluster_min_points:
            continue
        _, _, length = fit_line_pca(cluster)
        if cfg.cluster_min_length_m <= length <= cfg.cluster_max_length_m:
            output.append(cluster)
    return output


def fit_line_pca(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    center = np.mean(points, axis=0)
    centered = points - center
    covariance = centered.T @ centered / max(1, len(points) - 1)
    values, vectors = np.linalg.eigh(covariance)
    direction = vectors[:, int(np.argmax(values))]
    direction = direction / max(np.linalg.norm(direction), 1e-12)
    projection = centered @ direction
    length = float(np.max(projection) - np.min(projection))
    return center, direction, length


@dataclass(frozen=True)
class SlotEstimate:
    center: np.ndarray
    longitudinal: np.ndarray
    lateral: np.ndarray
    width: float
    confidence: float
    left_cluster: np.ndarray
    right_cluster: np.ndarray

    @property
    def yaw(self) -> float:
        return math.atan2(float(self.longitudinal[1]), float(self.longitudinal[0]))

    def to_world(self, longitudinal_m: float, lateral_m: float) -> np.ndarray:
        return self.center + longitudinal_m * self.longitudinal + lateral_m * self.lateral


class SlotDetector:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def detect(self, vehicle_points: np.ndarray) -> SlotEstimate:
        roi = vehicle_points[
            (vehicle_points[:, 0] >= self.cfg.detection_x_min_m)
            & (vehicle_points[:, 0] <= self.cfg.detection_x_max_m)
            & (np.abs(vehicle_points[:, 1]) <= self.cfg.detection_abs_y_max_m)
        ]
        if self.cfg.detection_rear_only:
            roi = roi[roi[:, 0] <= 0.5]
        clusters = cluster_ordered_scan(roi, self.cfg)
        if len(clusters) < 2:
            raise RuntimeError(f"Need two parked-car clusters; found {len(clusters)}")

        features = []
        for cluster in clusters:
            center, direction, length = fit_line_pca(cluster)
            features.append((cluster, center, direction, length))

        best: Optional[tuple[float, SlotEstimate]] = None
        expected = self.cfg.expected_slot_width_m
        parallel_limit = math.radians(self.cfg.car_pair_parallel_deg)

        for i in range(len(features)):
            c1, center1, d1, length1 = features[i]
            for j in range(i + 1, len(features)):
                c2, center2, d2, length2 = features[j]

                # Principal axes are sign-ambiguous.
                if float(np.dot(d1, d2)) < 0:
                    d2 = -d2
                angle_error = math.acos(float(np.clip(abs(np.dot(d1, d2)), 0.0, 1.0)))
                if angle_error > parallel_limit:
                    continue

                longitudinal = d1 + d2
                norm = float(np.linalg.norm(longitudinal))
                if norm < 1e-6:
                    continue
                longitudinal /= norm
                lateral = np.array([-longitudinal[1], longitudinal[0]])

                separation_vector = center2 - center1
                if float(np.dot(separation_vector, lateral)) < 0:
                    lateral = -lateral
                lateral_separation = abs(float(np.dot(separation_vector, lateral)))

                # Cluster centers are not necessarily the car centers. The pair
                # score only uses separation as a soft cue; inner edges below
                # provide the usable slot width.
                p1 = c1 @ lateral
                p2 = c2 @ lateral
                if float(np.mean(p1)) < float(np.mean(p2)):
                    left_cluster, right_cluster = c1, c2
                    inner_left = float(np.max(p1))
                    inner_right = float(np.min(p2))
                else:
                    left_cluster, right_cluster = c2, c1
                    inner_left = float(np.max(p2))
                    inner_right = float(np.min(p1))

                width = inner_right - inner_left
                if width <= 0:
                    continue
                if abs(width - expected) > self.cfg.slot_width_tolerance_m:
                    continue

                all_pair = np.vstack((left_cluster, right_cluster))
                longitudinal_coordinate = float(np.median(all_pair @ longitudinal))
                lateral_center_coordinate = 0.5 * (inner_left + inner_right)
                center = (
                    longitudinal_coordinate * longitudinal
                    + lateral_center_coordinate * lateral
                )

                width_error = abs(width - expected) / max(expected, 1e-6)
                longitudinal_misalignment = abs(
                    float(np.dot(center2 - center1, longitudinal))
                )
                score = (
                    2.2 * width_error
                    + 1.8 * angle_error
                    + 0.6 * longitudinal_misalignment
                    - 0.08 * (length1 + length2)
                )
                confidence = float(math.exp(-max(score, 0.0)))
                estimate = SlotEstimate(
                    center=center,
                    longitudinal=longitudinal,
                    lateral=lateral,
                    width=width,
                    confidence=confidence,
                    left_cluster=left_cluster,
                    right_cluster=right_cluster,
                )
                if best is None or score < best[0]:
                    best = (score, estimate)

        if best is None:
            raise RuntimeError(
                "No parallel parked-car pair matched the configured slot width"
            )

        slot = best[1]
        # Resolve the 180-degree direction ambiguity: the aisle/start vehicle is
        # normally in front of the slot. Choose +longitudinal toward the initial
        # vehicle origin so exit_forward_m points toward the aisle.
        vector_to_origin = -slot.center
        if float(np.dot(slot.longitudinal, vector_to_origin)) < 0:
            slot = SlotEstimate(
                center=slot.center,
                longitudinal=-slot.longitudinal,
                lateral=-slot.lateral,
                width=slot.width,
                confidence=slot.confidence,
                left_cluster=slot.right_cluster,
                right_cluster=slot.left_cluster,
            )
        return slot


# -----------------------------------------------------------------------------
# LiDAR-only scan matching
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ICPResult:
    transform_current_to_previous: Pose2D
    rmse: float
    pairs: int
    converged: bool


def best_fit_transform(source: np.ndarray, target: np.ndarray) -> Pose2D:
    source_center = np.mean(source, axis=0)
    target_center = np.mean(target, axis=0)
    source_zero = source - source_center
    target_zero = target - target_center
    h = source_zero.T @ target_zero
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T
    t = target_center - r @ source_center
    yaw = math.atan2(float(r[1, 0]), float(r[0, 0]))
    return Pose2D(float(t[0]), float(t[1]), yaw)


def icp_current_to_previous(
    previous: np.ndarray,
    current: np.ndarray,
    cfg: Config,
) -> ICPResult:
    """Find T such that T(current points) aligns with previous points."""
    if len(previous) < cfg.icp_min_pairs or len(current) < cfg.icp_min_pairs:
        return ICPResult(Pose2D(0.0, 0.0, 0.0), math.inf, 0, False)

    tree = cKDTree(previous)
    total = Pose2D(0.0, 0.0, 0.0)
    transformed = current.copy()
    previous_rmse = math.inf
    final_pairs = 0

    for _ in range(cfg.icp_max_iterations):
        distances, indices = tree.query(transformed, k=1)
        mask = distances <= cfg.icp_max_correspondence_m
        if int(np.count_nonzero(mask)) < cfg.icp_min_pairs:
            break

        candidate_distances = distances[mask]
        source = transformed[mask]
        target = previous[indices[mask]]

        keep_count = max(
            cfg.icp_min_pairs,
            int(len(candidate_distances) * cfg.icp_trim_fraction),
        )
        order = np.argsort(candidate_distances)[:keep_count]
        source = source[order]
        target = target[order]
        final_pairs = len(source)

        incremental = best_fit_transform(source, target)
        transformed = incremental.transform_points(transformed)
        total = incremental.compose(total)

        aligned_source = incremental.transform_points(source)
        residual = np.linalg.norm(aligned_source - target, axis=1)
        rmse = float(np.sqrt(np.mean(residual**2)))
        if abs(previous_rmse - rmse) < 1e-4:
            previous_rmse = rmse
            break
        previous_rmse = rmse

    translation = math.hypot(total.x, total.y)
    rotation_abs = abs(math.degrees(total.yaw))
    converged = (
        final_pairs >= cfg.icp_min_pairs
        and previous_rmse <= cfg.icp_max_rmse_m
        and translation <= cfg.icp_max_translation_per_scan_m
        and rotation_abs <= cfg.icp_max_rotation_per_scan_deg
    )
    return ICPResult(total, previous_rmse, final_pairs, converged)


# -----------------------------------------------------------------------------
# Hybrid A* planner
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class PathPoint:
    pose: Pose2D
    gear: int  # +1 forward, -1 reverse
    steering_norm: float


@dataclass(order=True)
class OpenItem:
    priority: float
    serial: int
    key: tuple[int, int, int, int] = field(compare=False)


@dataclass
class Node:
    pose: Pose2D
    gear: int
    steering_norm: float
    g: float
    parent: Optional[tuple[int, int, int, int]]


class HybridAStar:
    def __init__(self, cfg: Config, obstacles: np.ndarray) -> None:
        self.cfg = cfg
        self.obstacles = obstacles
        self.obstacle_tree = cKDTree(obstacles) if len(obstacles) else None
        self.max_steering = math.radians(cfg.max_steering_deg)
        self.yaw_resolution = math.radians(cfg.planner_yaw_resolution_deg)

    def _key(self, pose: Pose2D, gear: int) -> tuple[int, int, int, int]:
        return (
            int(round(pose.x / self.cfg.planner_xy_resolution_m)),
            int(round(pose.y / self.cfg.planner_xy_resolution_m)),
            int(round(wrap_angle(pose.yaw) / self.yaw_resolution)),
            1 if gear > 0 else -1,
        )

    def _collision(self, pose: Pose2D) -> bool:
        if self.obstacle_tree is None:
            return False

        front = self.cfg.vehicle_length_m - self.cfg.rear_axle_to_rear_m
        rear = self.cfg.rear_axle_to_rear_m
        half_width = self.cfg.vehicle_width_m / 2.0
        margin = self.cfg.safety_margin_m
        radius = math.hypot(max(front, rear) + margin, half_width + margin)
        candidates = self.obstacle_tree.query_ball_point([pose.x, pose.y], radius)
        if not candidates:
            return False
        points = self.obstacles[np.asarray(candidates, dtype=int)]
        local = (points - np.array([pose.x, pose.y])) @ rotation(-pose.yaw).T
        return bool(
            np.any(
                (local[:, 0] >= -rear - margin)
                & (local[:, 0] <= front + margin)
                & (np.abs(local[:, 1]) <= half_width + margin)
            )
        )

    def _simulate(
        self,
        pose: Pose2D,
        gear: int,
        steering_norm: float,
    ) -> Optional[Pose2D]:
        distance = self.cfg.planner_step_m * gear
        steering = steering_norm * self.max_steering
        substeps = max(3, int(abs(distance) / self.cfg.planner_collision_sample_m) + 1)
        ds = distance / substeps
        current = pose
        for _ in range(substeps):
            yaw_mid = current.yaw + 0.5 * ds * math.tan(steering) / self.cfg.wheelbase_m
            next_pose = Pose2D(
                x=current.x + ds * math.cos(yaw_mid),
                y=current.y + ds * math.sin(yaw_mid),
                yaw=wrap_angle(
                    current.yaw + ds * math.tan(steering) / self.cfg.wheelbase_m
                ),
            )
            if self._collision(next_pose):
                return None
            current = next_pose
        return current

    def _goal_reached(self, pose: Pose2D, goal: Pose2D) -> bool:
        return (
            math.hypot(pose.x - goal.x, pose.y - goal.y)
            <= self.cfg.planner_goal_xy_tolerance_m
            and abs(angle_diff(goal.yaw, pose.yaw))
            <= math.radians(self.cfg.planner_goal_yaw_tolerance_deg)
        )

    def _heuristic(self, pose: Pose2D, goal: Pose2D) -> float:
        distance = math.hypot(pose.x - goal.x, pose.y - goal.y)
        yaw_cost = 0.35 * self.cfg.wheelbase_m * abs(angle_diff(goal.yaw, pose.yaw))
        return distance + yaw_cost

    def plan(
        self,
        start: Pose2D,
        goal: Pose2D,
        prefer_final_reverse: bool,
    ) -> list[PathPoint]:
        if self._collision(start):
            raise RuntimeError("Planner start pose is in collision")
        if self._collision(goal):
            raise RuntimeError("Planner goal pose is in collision")

        margin = self.cfg.planner_bounds_margin_m
        if len(self.obstacles):
            min_xy = np.minimum(np.min(self.obstacles, axis=0), [start.x, start.y]) - margin
            max_xy = np.maximum(np.max(self.obstacles, axis=0), [start.x, start.y]) + margin
            min_xy = np.minimum(min_xy, [goal.x - margin, goal.y - margin])
            max_xy = np.maximum(max_xy, [goal.x + margin, goal.y + margin])
        else:
            min_xy = np.minimum([start.x, start.y], [goal.x, goal.y]) - margin
            max_xy = np.maximum([start.x, start.y], [goal.x, goal.y]) + margin

        start_key = self._key(start, 1)
        nodes: dict[tuple[int, int, int, int], Node] = {
            start_key: Node(start, 1, 0.0, 0.0, None)
        }
        best_cost = {start_key: 0.0}
        open_heap: list[OpenItem] = []
        serial_number = 0
        heapq.heappush(
            open_heap,
            OpenItem(self._heuristic(start, goal), serial_number, start_key),
        )

        steering_options = (-1.0, -0.5, 0.0, 0.5, 1.0)
        goal_key: Optional[tuple[int, int, int, int]] = None

        while open_heap and len(nodes) < self.cfg.planner_max_nodes:
            item = heapq.heappop(open_heap)
            current_key = item.key
            current = nodes[current_key]
            if current.g > best_cost.get(current_key, math.inf) + 1e-9:
                continue

            if self._goal_reached(current.pose, goal):
                # For parking, prefer ending in reverse but do not make the
                # planner impossible when the geometry requires correction.
                if not prefer_final_reverse or current.gear < 0:
                    goal_key = current_key
                    break

            for gear in (1, -1):
                for steering_norm in steering_options:
                    next_pose = self._simulate(current.pose, gear, steering_norm)
                    if next_pose is None:
                        continue
                    if not (
                        min_xy[0] <= next_pose.x <= max_xy[0]
                        and min_xy[1] <= next_pose.y <= max_xy[1]
                    ):
                        continue

                    move_cost = self.cfg.planner_step_m
                    if gear < 0:
                        move_cost *= self.cfg.planner_reverse_cost
                    if gear != current.gear:
                        move_cost += self.cfg.planner_gear_change_cost
                    move_cost += self.cfg.planner_steering_cost * abs(steering_norm)
                    move_cost += self.cfg.planner_steering_change_cost * abs(
                        steering_norm - current.steering_norm
                    )
                    next_g = current.g + move_cost
                    next_key = self._key(next_pose, gear)
                    if next_g >= best_cost.get(next_key, math.inf):
                        continue

                    best_cost[next_key] = next_g
                    nodes[next_key] = Node(
                        pose=next_pose,
                        gear=gear,
                        steering_norm=steering_norm,
                        g=next_g,
                        parent=current_key,
                    )
                    serial_number += 1
                    heapq.heappush(
                        open_heap,
                        OpenItem(
                            next_g + self._heuristic(next_pose, goal),
                            serial_number,
                            next_key,
                        ),
                    )

        if goal_key is None:
            raise RuntimeError(
                f"Hybrid A* failed after exploring {len(nodes)} states. "
                "Check footprint, slot geometry, steering limit, and margins."
            )

        reverse_path: list[PathPoint] = []
        key: Optional[tuple[int, int, int, int]] = goal_key
        while key is not None:
            node = nodes[key]
            reverse_path.append(PathPoint(node.pose, node.gear, node.steering_norm))
            key = node.parent
        reverse_path.reverse()
        return reverse_path


# -----------------------------------------------------------------------------
# Path tracking and mission state machine
# -----------------------------------------------------------------------------


class MissionState(Enum):
    INITIALIZING = auto()
    PARKING = auto()
    HOLDING = auto()
    EXITING = auto()
    DONE = auto()
    FAULT = auto()


@dataclass
class CommandChoice:
    steering: int
    drive_pwm: int
    cost: float
    predicted_pose: Pose2D


class RecedingHorizonController:
    def __init__(self, cfg: Config, obstacles: np.ndarray) -> None:
        self.cfg = cfg
        self.static_obstacles = obstacles.copy()
        self.obstacles = obstacles.copy()
        self.collision_checker = HybridAStar(cfg, self.obstacles)

    def update_dynamic_obstacles(self, current_world_points: np.ndarray) -> None:
        # The initial scan provides a stable planning map. The current scan is
        # added for safety so a newly appearing object can stop a command even
        # though it was absent during planning.
        if len(current_world_points):
            self.obstacles = voxel_downsample(
                np.vstack((self.static_obstacles, current_world_points)),
                self.cfg.voxel_size_m,
            )
        else:
            self.obstacles = self.static_obstacles.copy()
        self.collision_checker = HybridAStar(self.cfg, self.obstacles)

    def nearest_path_index(self, pose: Pose2D, path: Sequence[PathPoint]) -> int:
        distances = [
            (point.pose.x - pose.x) ** 2 + (point.pose.y - pose.y) ** 2
            for point in path
        ]
        return int(np.argmin(distances))

    def _target_index(
        self,
        pose: Pose2D,
        path: Sequence[PathPoint],
        nearest: int,
    ) -> int:
        accumulated = 0.0
        previous = path[nearest].pose
        for index in range(nearest + 1, len(path)):
            current = path[index].pose
            accumulated += math.hypot(current.x - previous.x, current.y - previous.y)
            if accumulated >= self.cfg.tracking_lookahead_m:
                return index
            previous = current
        return len(path) - 1

    def _direction_clear(self, current_vehicle_points: np.ndarray, gear: int) -> bool:
        if len(current_vehicle_points) == 0:
            return False
        front = self.cfg.vehicle_length_m - self.cfg.rear_axle_to_rear_m
        rear = self.cfg.rear_axle_to_rear_m
        half_width = self.cfg.vehicle_width_m / 2.0 + self.cfg.safety_margin_m
        if gear > 0:
            danger = (
                (current_vehicle_points[:, 0] > front)
                & (current_vehicle_points[:, 0] < front + self.cfg.stop_clearance_m)
                & (np.abs(current_vehicle_points[:, 1]) < half_width)
            )
        else:
            danger = (
                (current_vehicle_points[:, 0] < -rear)
                & (current_vehicle_points[:, 0] > -rear - self.cfg.stop_clearance_m)
                & (np.abs(current_vehicle_points[:, 1]) < half_width)
            )
        return not bool(np.any(danger))

    def choose(
        self,
        pose: Pose2D,
        path: Sequence[PathPoint],
        current_vehicle_points: np.ndarray,
    ) -> CommandChoice:
        nearest = self.nearest_path_index(pose, path)
        target_index = self._target_index(pose, path, nearest)
        target = path[target_index]
        gear = target.gear
        if not self._direction_clear(current_vehicle_points, gear):
            return CommandChoice(0, 0, math.inf, pose)
        speed = self.cfg.forward_speed_mps if gear > 0 else self.cfg.reverse_speed_mps
        drive_pwm = self.cfg.forward_pwm if gear > 0 else self.cfg.reverse_pwm

        best: Optional[CommandChoice] = None
        for steering_command in self.cfg.command_steering_levels:
            steering_angle = (
                steering_command / 1000.0 * math.radians(self.cfg.max_steering_deg)
            )
            distance = gear * speed * self.cfg.command_horizon_s
            yaw_delta = distance * math.tan(steering_angle) / self.cfg.wheelbase_m
            yaw_mid = pose.yaw + 0.5 * yaw_delta
            predicted = Pose2D(
                pose.x + distance * math.cos(yaw_mid),
                pose.y + distance * math.sin(yaw_mid),
                wrap_angle(pose.yaw + yaw_delta),
            )
            if self.collision_checker._collision(predicted):
                continue

            target_distance = math.hypot(
                predicted.x - target.pose.x,
                predicted.y - target.pose.y,
            )
            heading_error = abs(angle_diff(target.pose.yaw, predicted.yaw))
            final = path[-1].pose
            final_distance = math.hypot(predicted.x - final.x, predicted.y - final.y)
            steering_reference = target.steering_norm * 1000.0
            cost = (
                4.0 * target_distance**2
                + 0.8 * heading_error**2
                + 0.12 * final_distance**2
                + 0.0000008 * (steering_command - steering_reference) ** 2
            )
            candidate = CommandChoice(
                steering=steering_command,
                drive_pwm=drive_pwm,
                cost=cost,
                predicted_pose=predicted,
            )
            if best is None or candidate.cost < best.cost:
                best = candidate

        if best is None:
            return CommandChoice(0, 0, math.inf, pose)
        return best

    def path_error(self, pose: Pose2D, path: Sequence[PathPoint]) -> float:
        index = self.nearest_path_index(pose, path)
        point = path[index].pose
        return math.hypot(pose.x - point.x, pose.y - point.y)


class ParkingMission:
    def __init__(
        self,
        cfg: Config,
        lidar: LidarSource,
        arduino: ArduinoLink,
    ) -> None:
        self.cfg = cfg
        self.lidar = lidar
        self.arduino = arduino
        self.state = MissionState.INITIALIZING
        self.pose = Pose2D(0.0, 0.0, 0.0)
        self.previous_points: Optional[np.ndarray] = None
        self.initial_points: Optional[np.ndarray] = None
        self.slot: Optional[SlotEstimate] = None
        self.obstacles_world = np.empty((0, 2), dtype=np.float64)
        self.path: list[PathPoint] = []
        self.controller: Optional[RecedingHorizonController] = None
        self.goal_stable_since: Optional[float] = None
        self.hold_started: Optional[float] = None
        self.last_replan = 0.0
        self._stop_requested = False
        self._debug_poses: list[tuple[float, float, float, float]] = []
        self._debug_icp: list[tuple[float, float, int, bool]] = []

    def request_stop(self) -> None:
        self._stop_requested = True

    def _get_points(self) -> np.ndarray:
        stamp, scan = self.lidar.get_latest(timeout=1.0)
        age = time.monotonic() - stamp
        if age > self.cfg.max_scan_age_s:
            raise RuntimeError(f"LiDAR scan is stale: {age:.3f}s")
        points = polar_scan_to_vehicle_points(scan, self.cfg)
        if len(points) < self.cfg.scan_min_points:
            raise RuntimeError(
                f"Too few valid LiDAR points: {len(points)} < {self.cfg.scan_min_points}"
            )
        return points

    def initialize(self) -> None:
        print("Collecting stationary LiDAR scans...")
        scans: list[np.ndarray] = []
        for index in range(self.cfg.startup_scans):
            points = self._get_points()
            scans.append(points)
            print(f"  scan {index + 1}/{self.cfg.startup_scans}: {len(points)} points")

        # Use the last complete scan for ordered clustering. Multiple scans are
        # retained for optional debug, but combining them would destroy angle order.
        points = scans[-1]
        detector = SlotDetector(self.cfg)
        slot_vehicle = detector.detect(points)
        self.slot = slot_vehicle
        self.initial_points = points.copy()
        self.previous_points = points.copy()

        # Initial vehicle frame becomes the fixed world frame.
        # Use the complete stationary scan as the collision map, not only the
        # two detected inner vehicle faces. This preserves walls and the visible
        # outer portions of parked cars. Host-vehicle returns were removed in
        # polar_scan_to_vehicle_points().
        self.obstacles_world = points.copy()

        goal_point = slot_vehicle.to_world(
            self.cfg.goal_depth_from_car_centers_m,
            0.0,
        )
        # Vehicle faces toward the aisle when parked; reverse motion brings the
        # rear axle into the slot while preserving this heading.
        parking_goal = Pose2D(
            float(goal_point[0]),
            float(goal_point[1]),
            slot_vehicle.yaw,
        )

        print(
            "Detected slot: "
            f"width={slot_vehicle.width:.3f}m, "
            f"center=({slot_vehicle.center[0]:.3f}, {slot_vehicle.center[1]:.3f}), "
            f"yaw={math.degrees(slot_vehicle.yaw):.1f}deg, "
            f"confidence={slot_vehicle.confidence:.2f}"
        )
        print(
            "Parking goal rear axle: "
            f"({parking_goal.x:.3f}, {parking_goal.y:.3f}, "
            f"{math.degrees(parking_goal.yaw):.1f}deg)"
        )

        planner = HybridAStar(self.cfg, self.obstacles_world)
        self.path = planner.plan(
            start=self.pose,
            goal=parking_goal,
            prefer_final_reverse=True,
        )
        self.controller = RecedingHorizonController(self.cfg, self.obstacles_world)
        self.state = MissionState.PARKING
        print(f"Parking path generated: {len(self.path)} points")

    def _update_pose(self, current_points: np.ndarray) -> ICPResult:
        if self.previous_points is None:
            self.previous_points = current_points
            return ICPResult(Pose2D(0.0, 0.0, 0.0), 0.0, len(current_points), True)

        result = icp_current_to_previous(
            previous=self.previous_points,
            current=current_points,
            cfg=self.cfg,
        )
        if result.converged:
            # T maps current LiDAR/vehicle frame into the previous frame. It is
            # therefore the current vehicle pose increment expressed in the
            # previous frame and composes directly into the world pose.
            self.pose = self.pose.compose(result.transform_current_to_previous)
            self.previous_points = current_points
        self._debug_icp.append(
            (time.monotonic(), result.rmse, result.pairs, result.converged)
        )
        return result

    def _goal_reached(self, path: Sequence[PathPoint]) -> bool:
        goal = path[-1].pose
        return (
            math.hypot(self.pose.x - goal.x, self.pose.y - goal.y)
            <= self.cfg.planner_goal_xy_tolerance_m
            and abs(angle_diff(goal.yaw, self.pose.yaw))
            <= math.radians(self.cfg.planner_goal_yaw_tolerance_deg)
        )

    def _plan_exit(self) -> None:
        assert self.slot is not None
        exit_point = self.slot.to_world(
            self.cfg.exit_forward_m,
            self.cfg.exit_lateral_m,
        )
        exit_goal = Pose2D(
            float(exit_point[0]),
            float(exit_point[1]),
            wrap_angle(self.slot.yaw + math.radians(self.cfg.exit_yaw_offset_deg)),
        )
        planner = HybridAStar(self.cfg, self.obstacles_world)
        self.path = planner.plan(self.pose, exit_goal, prefer_final_reverse=False)
        self.controller = RecedingHorizonController(self.cfg, self.obstacles_world)
        self.state = MissionState.EXITING
        self.goal_stable_since = None
        print(f"Exit path generated: {len(self.path)} points")

    def _replan_current_goal(self) -> None:
        if not self.path:
            return
        now = time.monotonic()
        if now - self.last_replan < self.cfg.replan_interval_s:
            return
        self.last_replan = now
        goal = self.path[-1].pose
        planner = HybridAStar(self.cfg, self.obstacles_world)
        prefer_reverse = self.state == MissionState.PARKING
        self.path = planner.plan(self.pose, goal, prefer_final_reverse=prefer_reverse)
        self.controller = RecedingHorizonController(self.cfg, self.obstacles_world)
        print(f"Replanned from LiDAR pose: {len(self.path)} points")

    def run(self) -> None:
        self.initialize()
        print("Ready. Keep the vehicle lifted for the first test.")
        input("Press ENTER to arm and begin; Ctrl+C performs emergency stop... ")

        period = 1.0 / self.cfg.control_hz
        last_print = 0.0
        while not self._stop_requested and self.state not in (
            MissionState.DONE,
            MissionState.FAULT,
        ):
            cycle_started = time.monotonic()
            try:
                current_points = self._get_points()
                icp = self._update_pose(current_points)
                if not icp.converged:
                    self.arduino.stop_vehicle()
                    print(
                        f"ICP rejected: rmse={icp.rmse:.3f}, pairs={icp.pairs}. "
                        "Vehicle stopped; waiting for the next scan.",
                        file=sys.stderr,
                    )
                    time.sleep(period)
                    continue

                self._debug_poses.append(
                    (time.monotonic(), self.pose.x, self.pose.y, self.pose.yaw)
                )

                now = time.monotonic()
                if self.state in (MissionState.PARKING, MissionState.EXITING):
                    assert self.controller is not None
                    path_error = self.controller.path_error(self.pose, self.path)
                    if path_error > self.cfg.max_path_error_m:
                        self.arduino.stop_vehicle()
                        self._replan_current_goal()
                    elif self._goal_reached(self.path):
                        self.arduino.stop_vehicle()
                        if self.goal_stable_since is None:
                            self.goal_stable_since = now
                        elif now - self.goal_stable_since >= self.cfg.goal_stable_s:
                            if self.state == MissionState.PARKING:
                                self.state = MissionState.HOLDING
                                self.hold_started = now
                                print(f"Parking pose reached. Holding for {self.cfg.goal_hold_s:.1f} seconds...")
                            else:
                                self.state = MissionState.DONE
                                print("Exit goal reached. Mission complete.")
                    else:
                        self.goal_stable_since = None
                        current_world_points = self.pose.transform_points(current_points)
                        self.controller.update_dynamic_obstacles(current_world_points)
                        choice = self.controller.choose(
                            self.pose,
                            self.path,
                            current_vehicle_points=current_points,
                        )
                        if choice.drive_pwm == 0:
                            self.arduino.stop_vehicle()
                            self._replan_current_goal()
                        else:
                            self.arduino.send_control(choice.steering, choice.drive_pwm)

                elif self.state == MissionState.HOLDING:
                    self.arduino.stop_vehicle()
                    assert self.hold_started is not None
                    if now - self.hold_started >= self.cfg.goal_hold_s:
                        self._plan_exit()

                if now - last_print >= 0.5:
                    last_print = now
                    print(
                        f"state={self.state.name:<10} "
                        f"pose=({self.pose.x:+.2f},{self.pose.y:+.2f},"
                        f"{math.degrees(self.pose.yaw):+.1f}deg) "
                        f"icp={icp.rmse:.3f}m/{icp.pairs}"
                    )

            except Exception as exc:
                self.state = MissionState.FAULT
                self.arduino.stop_vehicle()
                print(f"MISSION FAULT: {exc}", file=sys.stderr)
                raise
            finally:
                elapsed = time.monotonic() - cycle_started
                if elapsed < period:
                    time.sleep(period - elapsed)

    def save_debug(self) -> None:
        path = Path(self.cfg.save_debug_npz)
        try:
            np.savez_compressed(
                path,
                poses=np.asarray(self._debug_poses, dtype=np.float64),
                icp=np.asarray(self._debug_icp, dtype=object),
                initial_points=(
                    self.initial_points
                    if self.initial_points is not None
                    else np.empty((0, 2))
                ),
                obstacles=self.obstacles_world,
                path=np.asarray(
                    [
                        (p.pose.x, p.pose.y, p.pose.yaw, p.gear, p.steering_norm)
                        for p in self.path
                    ],
                    dtype=np.float64,
                ),
            )
            print(f"Debug data saved: {path}")
        except Exception as exc:
            print(f"Could not save debug data: {exc}", file=sys.stderr)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LiDAR-only autonomous parking using the existing Arduino C command"
    )
    parser.add_argument("--config", type=Path, default=Path("parking_config.json"))
    parser.add_argument("--arduino", help="Override Arduino COM port")
    parser.add_argument("--lidar", help="Override LiDAR COM port")
    parser.add_argument("--dry-run", action="store_true", help="Do not command motors")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = Config.load(args.config)
    if args.arduino:
        cfg.arduino_port = args.arduino
    if args.lidar:
        cfg.lidar_port = args.lidar
    if args.dry_run:
        cfg.dry_run = True

    arduino = ArduinoLink(cfg.arduino_port, cfg.arduino_baud, cfg.dry_run)
    lidar: LidarSource = RPLidarSource(cfg.lidar_port, cfg.lidar_baud)
    mission = ParkingMission(cfg, lidar, arduino)

    def emergency_handler(signum: int, frame: object) -> None:
        del signum, frame
        mission.request_stop()
        arduino.stop_vehicle()

    signal.signal(signal.SIGINT, emergency_handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, emergency_handler)

    try:
        arduino.open()
        lidar.start()
        mission.run()
        return 0 if mission.state == MissionState.DONE else 2
    except KeyboardInterrupt:
        print("Emergency stop requested.")
        return 130
    except Exception as exc:
        print(f"Fatal error: {exc}", file=sys.stderr)
        return 1
    finally:
        arduino.stop_vehicle()
        lidar.close()
        arduino.close()
        mission.save_debug()


if __name__ == "__main__":
    raise SystemExit(main())
