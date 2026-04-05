"""Tests for scene_db.sequence_analysis."""

import math
import sqlite3
from datetime import datetime

import pytest

from scene_db.db import SCHEMA_SQL, insert_scene_chunks
from scene_db.models import SceneChunk
from scene_db.sequence_analysis import (
    _haversine,
    _euclidean_2d,
    _detect_loop_from_positions,
    _lat_lon_to_local,
    analyze_sequences,
)


@pytest.fixture
def conn():
    """In-memory SQLite connection with schema applied."""
    c = sqlite3.connect(":memory:")
    c.executescript(SCHEMA_SQL)
    yield c
    c.close()


def _make_chunk(
    chunk_id="test_000",
    dataset_name="test",
    sequence_id="seq",
    chunk_index=0,
    avg_speed_kmh=30.0,
    distance_m=50.0,
    max_decel_ms2=1.0,
    max_yaw_rate_degs=5.0,
    start_time=None,
    end_time=None,
):
    t0 = start_time or datetime(2024, 1, 1, 0, 0, 0)
    t1 = end_time or datetime(2024, 1, 1, 0, 0, 5)
    return SceneChunk(
        id=chunk_id,
        dataset_name=dataset_name,
        sequence_id=sequence_id,
        chunk_index=chunk_index,
        start_time=t0,
        end_time=t1,
        start_frame=chunk_index * 50,
        end_frame=chunk_index * 50 + 49,
        avg_speed_kmh=avg_speed_kmh,
        distance_m=distance_m,
        max_decel_ms2=max_decel_ms2,
        max_yaw_rate_degs=max_yaw_rate_degs,
    )


class TestHaversine:
    def test_same_point(self):
        assert _haversine(35.0, 139.0, 35.0, 139.0) == 0.0

    def test_known_distance(self):
        # Tokyo (35.6762, 139.6503) to Yokohama (35.4437, 139.6380)
        # Approx ~26 km
        d = _haversine(35.6762, 139.6503, 35.4437, 139.6380)
        assert 25_000 < d < 27_000

    def test_short_distance(self):
        # ~111 m per 0.001 degree latitude
        d = _haversine(35.0, 139.0, 35.001, 139.0)
        assert 100 < d < 120


class TestEuclidean2d:
    def test_zero_distance(self):
        assert _euclidean_2d(0, 0, 0, 0) == 0.0

    def test_known_triangle(self):
        assert _euclidean_2d(0, 0, 3, 4) == pytest.approx(5.0)

    def test_negative_coords(self):
        assert _euclidean_2d(-1, -1, 2, 3) == pytest.approx(5.0)


class TestDetectLoopFromPositions:
    def test_single_point(self):
        has_loop, dist, revisits = _detect_loop_from_positions([(0, 0)])
        assert has_loop is False
        assert dist == 0.0
        assert revisits == 0

    def test_loop_closed(self):
        # Create a square loop returning to start
        positions = [(0, 0), (100, 0), (100, 100), (0, 100), (2, 2)]
        has_loop, dist, revisits = _detect_loop_from_positions(positions)
        assert has_loop is True
        assert dist < 10.0

    def test_no_loop(self):
        positions = [(0, 0), (100, 0), (200, 0), (300, 0)]
        has_loop, dist, revisits = _detect_loop_from_positions(positions)
        assert has_loop is False
        assert dist > 10.0

    def test_revisit_detection(self):
        # Create a trajectory that revisits a location after sufficient gap
        # Needs enough points to exceed min_gap
        positions = []
        # Go out
        for i in range(30):
            positions.append((i * 5.0, 0.0))
        # Come back through the same area
        for i in range(30, 0, -1):
            positions.append((i * 5.0, 1.0))  # slightly offset in y
        has_loop, dist, revisits = _detect_loop_from_positions(positions, threshold_m=10.0)
        assert revisits > 0


class TestLatLonToLocal:
    def test_same_point(self):
        x, y = _lat_lon_to_local(35.0, 139.0, 35.0, 139.0)
        assert x == pytest.approx(0.0)
        assert y == pytest.approx(0.0)

    def test_offset_north(self):
        # Moving north should increase y
        x, y = _lat_lon_to_local(35.001, 139.0, 35.0, 139.0)
        assert abs(x) < 1.0  # no east-west movement
        assert y > 100  # ~111m per 0.001 deg

    def test_offset_east(self):
        # Moving east should increase x
        x, y = _lat_lon_to_local(35.0, 139.001, 35.0, 139.0)
        assert x > 50  # positive east offset
        assert abs(y) < 1.0  # no north-south movement


class TestAnalyzeSequences:
    def test_single_sequence(self, conn):
        chunks = [
            _make_chunk(
                chunk_id="t0",
                chunk_index=0,
                start_time=datetime(2024, 1, 1, 0, 0, 0),
                end_time=datetime(2024, 1, 1, 0, 0, 5),
                avg_speed_kmh=30.0,
                distance_m=50.0,
            ),
            _make_chunk(
                chunk_id="t1",
                chunk_index=1,
                start_time=datetime(2024, 1, 1, 0, 0, 5),
                end_time=datetime(2024, 1, 1, 0, 0, 10),
                avg_speed_kmh=40.0,
                distance_m=60.0,
            ),
        ]
        insert_scene_chunks(conn, chunks)
        results = analyze_sequences(conn)
        assert len(results) == 1
        info = results[0]
        assert info.dataset_name == "test"
        assert info.sequence_id == "seq"
        assert info.total_chunks == 2
        assert info.total_distance_m == pytest.approx(110.0)
        assert info.avg_speed_kmh == pytest.approx(35.0)
        assert info.max_speed_kmh == pytest.approx(40.0)
        assert info.duration_sec == pytest.approx(10.0)

    def test_multiple_sequences(self, conn):
        chunks = [
            _make_chunk(chunk_id="a0", dataset_name="ds1", sequence_id="s1", chunk_index=0),
            _make_chunk(chunk_id="b0", dataset_name="ds1", sequence_id="s2", chunk_index=0),
            _make_chunk(chunk_id="c0", dataset_name="ds2", sequence_id="s1", chunk_index=0),
        ]
        insert_scene_chunks(conn, chunks)
        results = analyze_sequences(conn)
        assert len(results) == 3

    def test_empty_db(self, conn):
        results = analyze_sequences(conn)
        assert results == []
