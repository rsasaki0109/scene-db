"""Feature extraction and caption generation."""

import math
from dataclasses import dataclass

from scene_db.models import OxtsRecord


@dataclass
class SceneFeatures:
    """Computed features for a scene chunk."""

    avg_speed_kmh: float = 0.0
    distance_m: float = 0.0
    max_accel_ms2: float = 0.0
    max_decel_ms2: float = 0.0
    avg_yaw_rate_degs: float = 0.0
    max_yaw_rate_degs: float = 0.0


def _speed(r: OxtsRecord) -> float:
    return math.sqrt(r.vf**2 + r.vl**2)


def compute_avg_speed_kmh(records: list[OxtsRecord]) -> float:
    """Compute average speed in km/h from forward and lateral velocities."""
    if not records:
        return 0.0
    total = sum(_speed(r) for r in records)
    return (total / len(records)) * 3.6  # m/s -> km/h


def compute_distance_m(records: list[OxtsRecord]) -> float:
    """Compute total distance traveled using trapezoidal integration."""
    if len(records) < 2:
        return 0.0
    distance = 0.0
    for i in range(1, len(records)):
        dt = (records[i].timestamp - records[i - 1].timestamp).total_seconds()
        if dt <= 0:
            continue
        distance += 0.5 * (_speed(records[i - 1]) + _speed(records[i])) * dt
    return distance


def compute_acceleration(records: list[OxtsRecord]) -> tuple[float, float]:
    """Compute max acceleration and max deceleration (m/s^2).

    Returns (max_accel, max_decel) where max_decel is positive for braking.
    """
    if len(records) < 2:
        return 0.0, 0.0
    max_accel = 0.0
    max_decel = 0.0
    for i in range(1, len(records)):
        dt = (records[i].timestamp - records[i - 1].timestamp).total_seconds()
        if dt <= 0:
            continue
        dv = _speed(records[i]) - _speed(records[i - 1])
        accel = dv / dt
        if accel > max_accel:
            max_accel = accel
        if accel < 0 and (-accel) > max_decel:
            max_decel = -accel
    return max_accel, max_decel


def compute_yaw_rate(records: list[OxtsRecord]) -> tuple[float, float]:
    """Compute average and max yaw rate in deg/s.

    Returns (avg_yaw_rate, max_yaw_rate).
    """
    if len(records) < 2:
        return 0.0, 0.0
    yaw_rates = []
    for i in range(1, len(records)):
        dt = (records[i].timestamp - records[i - 1].timestamp).total_seconds()
        if dt <= 0:
            continue
        # Normalize yaw difference to [-pi, pi]
        dyaw = records[i].yaw - records[i - 1].yaw
        dyaw = math.atan2(math.sin(dyaw), math.cos(dyaw))
        yaw_rates.append(abs(math.degrees(dyaw / dt)))
    if not yaw_rates:
        return 0.0, 0.0
    return sum(yaw_rates) / len(yaw_rates), max(yaw_rates)


def extract_features(records: list[OxtsRecord]) -> SceneFeatures:
    """Extract all features from a chunk's records."""
    avg_speed = compute_avg_speed_kmh(records)
    distance = compute_distance_m(records)
    max_accel, max_decel = compute_acceleration(records)
    avg_yaw_rate, max_yaw_rate = compute_yaw_rate(records)
    return SceneFeatures(
        avg_speed_kmh=avg_speed,
        distance_m=distance,
        max_accel_ms2=max_accel,
        max_decel_ms2=max_decel,
        avg_yaw_rate_degs=avg_yaw_rate,
        max_yaw_rate_degs=max_yaw_rate,
    )


def generate_caption(
    avg_speed_kmh: float,
    distance_m: float,
    max_decel_ms2: float = 0.0,
    avg_yaw_rate_degs: float = 0.0,
    max_yaw_rate_degs: float = 0.0,
) -> str:
    """Generate a rule-based caption from features."""
    # Motion state
    if avg_speed_kmh < 1.0:
        motion = "vehicle stationary"
    elif avg_speed_kmh < 15.0:
        motion = "vehicle moving slowly"
    elif avg_speed_kmh < 50.0:
        motion = "vehicle moving forward"
    else:
        motion = "vehicle moving at high speed"

    parts = [motion, f"{avg_speed_kmh:.0f} km/h", f"traveled {distance_m:.1f} m"]

    # Turning behavior
    if max_yaw_rate_degs > 30.0:
        parts.append("sharp turn")
    elif avg_yaw_rate_degs > 10.0:
        parts.append("turning")
    elif avg_yaw_rate_degs > 3.0:
        parts.append("gentle curve")

    # Braking behavior
    if max_decel_ms2 > 3.0:
        parts.append("hard braking")
    elif max_decel_ms2 > 1.0:
        parts.append("braking")

    return ", ".join(parts)
