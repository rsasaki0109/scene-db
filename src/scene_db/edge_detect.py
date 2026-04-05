"""Automatic edge case detection for localization and perception."""

import sqlite3
from dataclasses import dataclass

from scene_db.db import list_all_scenes
from scene_db.models import SceneChunk


@dataclass
class EdgeCase:
    """A detected edge case with reason and severity."""

    scene: SceneChunk
    category: str  # "localization", "perception", "both"
    reason: str
    severity: str  # "critical", "warning", "info"
    score: float  # 0-1, higher = more extreme


def detect_edge_cases(conn: sqlite3.Connection) -> list[EdgeCase]:
    """Detect edge cases from all scenes in the database."""
    scenes = list_all_scenes(conn)
    if not scenes:
        return []

    # Compute statistics for thresholds
    speeds = [s.avg_speed_kmh for s in scenes if s.avg_speed_kmh > 0]
    decels = [s.max_decel_ms2 for s in scenes if s.max_decel_ms2 > 0]
    yaws = [s.max_yaw_rate_degs for s in scenes if s.max_yaw_rate_degs > 0]

    speed_p90 = _percentile(speeds, 0.90) if speeds else 50.0
    decel_p90 = _percentile(decels, 0.90) if decels else 2.0
    yaw_p90 = _percentile(yaws, 0.90) if yaws else 10.0

    edge_cases = []

    for s in scenes:
        cases = []

        # --- Localization edge cases ---

        # High yaw rate (turning stress)
        if s.max_yaw_rate_degs > 20.0:
            cases.append(EdgeCase(
                scene=s, category="localization",
                reason=f"High yaw rate ({s.max_yaw_rate_degs:.1f} deg/s) - IMU heading drift risk",
                severity="critical",
                score=min(s.max_yaw_rate_degs / 30.0, 1.0),
            ))
        elif s.max_yaw_rate_degs > 10.0:
            cases.append(EdgeCase(
                scene=s, category="localization",
                reason=f"Elevated yaw rate ({s.max_yaw_rate_degs:.1f} deg/s) - heading estimate stress",
                severity="warning",
                score=s.max_yaw_rate_degs / 30.0,
            ))

        # High speed (GPS latency becomes significant)
        if s.avg_speed_kmh > 60.0:
            latency_error_m = (s.avg_speed_kmh / 3.6) * 0.05  # 50ms latency
            cases.append(EdgeCase(
                scene=s, category="localization",
                reason=f"High speed ({s.avg_speed_kmh:.0f} km/h) - 50ms GPS latency = {latency_error_m:.1f}m error",
                severity="critical" if s.avg_speed_kmh > 80 else "warning",
                score=min(s.avg_speed_kmh / 100.0, 1.0),
            ))

        # Near-zero speed (GPS noise dominates, wheel odometry unreliable)
        if 0 < s.avg_speed_kmh < 3.0 and s.distance_m > 0:
            cases.append(EdgeCase(
                scene=s, category="localization",
                reason=f"Near-zero speed ({s.avg_speed_kmh:.1f} km/h) - GPS noise > motion signal",
                severity="warning",
                score=0.6,
            ))

        # Stationary to moving transition (GNSS reacquisition)
        if s.avg_speed_kmh < 1.0 and s.max_accel_ms2 > 0.5:
            cases.append(EdgeCase(
                scene=s, category="localization",
                reason=f"Start from stop (accel {s.max_accel_ms2:.1f} m/s²) - GNSS reacquisition jump risk",
                severity="warning",
                score=0.5,
            ))

        # Combined yaw + decel (multi-axis stress)
        if s.max_yaw_rate_degs > 10.0 and s.max_decel_ms2 > 1.0:
            combined = (s.max_yaw_rate_degs / 30.0 + s.max_decel_ms2 / 3.0) / 2
            cases.append(EdgeCase(
                scene=s, category="both",
                reason=(f"Yaw {s.max_yaw_rate_degs:.1f} deg/s + decel {s.max_decel_ms2:.1f} m/s² "
                        f"- multi-axis sensor stress"),
                severity="critical",
                score=min(combined, 1.0),
            ))

        # --- Perception edge cases ---

        # Hard braking (pitch shift affects LiDAR/camera FOV)
        if s.max_decel_ms2 > 3.0:
            cases.append(EdgeCase(
                scene=s, category="perception",
                reason=f"Hard braking ({s.max_decel_ms2:.1f} m/s²) - pitch shift affects LiDAR/camera FOV",
                severity="critical",
                score=min(s.max_decel_ms2 / 5.0, 1.0),
            ))
        elif s.max_decel_ms2 > 1.5:
            cases.append(EdgeCase(
                scene=s, category="perception",
                reason=f"Braking ({s.max_decel_ms2:.1f} m/s²) - sensor geometry change",
                severity="warning",
                score=s.max_decel_ms2 / 5.0,
            ))

        # Speed transition (tracking handoff)
        if s.max_decel_ms2 > 1.0 and s.avg_speed_kmh < 5.0:
            cases.append(EdgeCase(
                scene=s, category="perception",
                reason=f"Decel to near-stop ({s.avg_speed_kmh:.0f} km/h, decel {s.max_decel_ms2:.1f}) - tracking handoff zone",
                severity="warning",
                score=0.5 + s.max_decel_ms2 / 10.0,
            ))

        # Statistical outliers (above 90th percentile)
        if s.max_yaw_rate_degs > yaw_p90 and s.max_yaw_rate_degs > 5.0:
            is_dup = any(c.reason.startswith("High yaw") or c.reason.startswith("Elevated yaw")
                        for c in cases)
            if not is_dup:
                cases.append(EdgeCase(
                    scene=s, category="localization",
                    reason=f"Yaw rate outlier ({s.max_yaw_rate_degs:.1f} deg/s > p90 {yaw_p90:.1f})",
                    severity="info",
                    score=0.4,
                ))

        if s.max_decel_ms2 > decel_p90 and s.max_decel_ms2 > 0.5:
            is_dup = any("braking" in c.reason.lower() or "decel" in c.reason.lower()
                         for c in cases)
            if not is_dup:
                cases.append(EdgeCase(
                    scene=s, category="perception",
                    reason=f"Decel outlier ({s.max_decel_ms2:.1f} m/s² > p90 {decel_p90:.1f})",
                    severity="info",
                    score=0.3,
                ))

        edge_cases.extend(cases)

    # Sort by severity then score
    severity_order = {"critical": 0, "warning": 1, "info": 2}
    edge_cases.sort(key=lambda e: (severity_order[e.severity], -e.score))

    return edge_cases


def _percentile(values: list[float], p: float) -> float:
    """Compute the p-th percentile of a sorted list."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = int(len(sorted_vals) * p)
    idx = min(idx, len(sorted_vals) - 1)
    return sorted_vals[idx]
