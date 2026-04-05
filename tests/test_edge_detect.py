"""Tests for scene_db.edge_detect."""

import sqlite3
from datetime import datetime

import pytest

from scene_db.db import SCHEMA_SQL, insert_scene_chunks
from scene_db.edge_detect import detect_edge_cases, EdgeCase
from scene_db.models import SceneChunk


@pytest.fixture
def conn():
    """In-memory SQLite connection with schema applied."""
    c = sqlite3.connect(":memory:")
    c.executescript(SCHEMA_SQL)
    yield c
    c.close()


def _make_chunk(
    chunk_id="test_000",
    avg_speed_kmh=0.0,
    distance_m=0.0,
    max_accel_ms2=0.0,
    max_decel_ms2=0.0,
    avg_yaw_rate_degs=0.0,
    max_yaw_rate_degs=0.0,
    caption="",
    chunk_index=0,
):
    t0 = datetime(2024, 1, 1, 0, 0, 0)
    t1 = datetime(2024, 1, 1, 0, 0, 5)
    return SceneChunk(
        id=chunk_id,
        dataset_name="test",
        sequence_id="seq",
        chunk_index=chunk_index,
        start_time=t0,
        end_time=t1,
        start_frame=0,
        end_frame=49,
        avg_speed_kmh=avg_speed_kmh,
        distance_m=distance_m,
        max_accel_ms2=max_accel_ms2,
        max_decel_ms2=max_decel_ms2,
        avg_yaw_rate_degs=avg_yaw_rate_degs,
        max_yaw_rate_degs=max_yaw_rate_degs,
        caption=caption,
    )


class TestDetectEdgeCases:
    def test_empty_scenes(self, conn):
        result = detect_edge_cases(conn)
        assert result == []

    def test_high_yaw_rate_critical(self, conn):
        chunk = _make_chunk(chunk_id="yaw_high", max_yaw_rate_degs=25.0)
        insert_scene_chunks(conn, [chunk])
        results = detect_edge_cases(conn)
        yaw_cases = [e for e in results if "High yaw rate" in e.reason]
        assert len(yaw_cases) == 1
        assert yaw_cases[0].severity == "critical"
        assert yaw_cases[0].category == "localization"

    def test_elevated_yaw_rate_warning(self, conn):
        chunk = _make_chunk(chunk_id="yaw_med", max_yaw_rate_degs=15.0)
        insert_scene_chunks(conn, [chunk])
        results = detect_edge_cases(conn)
        yaw_cases = [e for e in results if "Elevated yaw rate" in e.reason]
        assert len(yaw_cases) == 1
        assert yaw_cases[0].severity == "warning"

    def test_high_speed_critical(self, conn):
        chunk = _make_chunk(chunk_id="fast", avg_speed_kmh=90.0)
        insert_scene_chunks(conn, [chunk])
        results = detect_edge_cases(conn)
        speed_cases = [e for e in results if "High speed" in e.reason]
        assert len(speed_cases) == 1
        assert speed_cases[0].severity == "critical"

    def test_high_speed_warning(self, conn):
        chunk = _make_chunk(chunk_id="fast_w", avg_speed_kmh=70.0)
        insert_scene_chunks(conn, [chunk])
        results = detect_edge_cases(conn)
        speed_cases = [e for e in results if "High speed" in e.reason]
        assert len(speed_cases) == 1
        assert speed_cases[0].severity == "warning"

    def test_near_zero_speed_warning(self, conn):
        chunk = _make_chunk(chunk_id="slow", avg_speed_kmh=2.0, distance_m=5.0)
        insert_scene_chunks(conn, [chunk])
        results = detect_edge_cases(conn)
        slow_cases = [e for e in results if "Near-zero speed" in e.reason]
        assert len(slow_cases) == 1
        assert slow_cases[0].severity == "warning"

    def test_combined_yaw_decel_critical(self, conn):
        chunk = _make_chunk(
            chunk_id="combo",
            max_yaw_rate_degs=15.0,
            max_decel_ms2=2.0,
        )
        insert_scene_chunks(conn, [chunk])
        results = detect_edge_cases(conn)
        combo_cases = [e for e in results if "multi-axis" in e.reason]
        assert len(combo_cases) == 1
        assert combo_cases[0].severity == "critical"
        assert combo_cases[0].category == "both"

    def test_hard_braking_critical_perception(self, conn):
        chunk = _make_chunk(chunk_id="brake", max_decel_ms2=4.0)
        insert_scene_chunks(conn, [chunk])
        results = detect_edge_cases(conn)
        brake_cases = [e for e in results if "Hard braking" in e.reason]
        assert len(brake_cases) == 1
        assert brake_cases[0].severity == "critical"
        assert brake_cases[0].category == "perception"

    def test_moderate_braking_warning(self, conn):
        chunk = _make_chunk(chunk_id="brake_w", max_decel_ms2=2.0)
        insert_scene_chunks(conn, [chunk])
        results = detect_edge_cases(conn)
        brake_cases = [e for e in results if "Braking" in e.reason and "sensor geometry" in e.reason]
        assert len(brake_cases) == 1
        assert brake_cases[0].severity == "warning"

    def test_lidar_degeneration_warning(self, conn):
        chunk = _make_chunk(
            chunk_id="degen",
            avg_speed_kmh=3.0,
            max_yaw_rate_degs=1.0,
            max_decel_ms2=0.2,
            distance_m=10.0,
        )
        insert_scene_chunks(conn, [chunk])
        results = detect_edge_cases(conn)
        degen_cases = [e for e in results if "LiDAR degeneration" in e.reason]
        assert len(degen_cases) == 1
        assert degen_cases[0].severity == "warning"
        assert degen_cases[0].category == "localization"

    def test_imu_drift_critical(self, conn):
        chunk = _make_chunk(chunk_id="drift", avg_speed_kmh=250.0)
        insert_scene_chunks(conn, [chunk])
        results = detect_edge_cases(conn)
        drift_cases = [e for e in results if "IMU drift" in e.reason]
        assert len(drift_cases) == 1
        assert drift_cases[0].severity == "critical"
        assert drift_cases[0].score == 1.0

    def test_results_sorted_by_severity_then_score(self, conn):
        chunks = [
            _make_chunk(chunk_id="c0", chunk_index=0, max_yaw_rate_degs=25.0),  # critical
            _make_chunk(chunk_id="c1", chunk_index=1, avg_speed_kmh=2.0, distance_m=5.0),  # warning
        ]
        insert_scene_chunks(conn, chunks)
        results = detect_edge_cases(conn)
        assert len(results) >= 2
        # All criticals before all warnings
        severity_values = [e.severity for e in results]
        first_warning = severity_values.index("warning") if "warning" in severity_values else len(severity_values)
        for i in range(first_warning):
            assert severity_values[i] == "critical"
