"""Tests for scene_db.visualize."""

from datetime import datetime
from pathlib import Path

import pytest

plt = pytest.importorskip("matplotlib")

from scene_db.edge_detect import EdgeCase
from scene_db.models import SceneChunk
from scene_db.visualize import (
    plot_feature_histograms,
    plot_trajectory,
    plot_edge_case_summary,
)


def _make_chunk(
    chunk_id="test_000",
    avg_speed_kmh=30.0,
    max_decel_ms2=1.0,
    max_yaw_rate_degs=5.0,
    caption="vehicle moving forward",
):
    t0 = datetime(2024, 1, 1, 0, 0, 0)
    t1 = datetime(2024, 1, 1, 0, 0, 5)
    return SceneChunk(
        id=chunk_id,
        dataset_name="test",
        sequence_id="seq",
        chunk_index=0,
        start_time=t0,
        end_time=t1,
        start_frame=0,
        end_frame=49,
        avg_speed_kmh=avg_speed_kmh,
        max_decel_ms2=max_decel_ms2,
        max_yaw_rate_degs=max_yaw_rate_degs,
        caption=caption,
    )


def _make_edge_case(severity="critical", category="localization"):
    chunk = _make_chunk()
    return EdgeCase(
        scene=chunk,
        category=category,
        reason="Test edge case",
        severity=severity,
        score=0.8,
    )


class TestPlotFeatureHistograms:
    def test_creates_png(self, tmp_path):
        scenes = [
            _make_chunk(chunk_id=f"s{i}", avg_speed_kmh=10.0 * i)
            for i in range(5)
        ]
        out = tmp_path / "hist.png"
        result = plot_feature_histograms(scenes, out)
        assert Path(result).exists()
        assert Path(result).stat().st_size > 0

    def test_empty_scenes(self, tmp_path):
        out = tmp_path / "hist_empty.png"
        result = plot_feature_histograms([], out)
        assert Path(result).exists()


class TestPlotTrajectory:
    def test_creates_png(self, tmp_path):
        positions = [(i * 10.0, i * 5.0) for i in range(20)]
        out = tmp_path / "traj.png"
        result = plot_trajectory(positions, out, title="Test Trajectory")
        assert Path(result).exists()
        assert Path(result).stat().st_size > 0

    def test_loop_trajectory(self, tmp_path):
        # Create a closed loop
        import math
        positions = [
            (100 * math.cos(t), 100 * math.sin(t))
            for t in [i * 2 * math.pi / 50 for i in range(51)]
        ]
        out = tmp_path / "loop.png"
        result = plot_trajectory(positions, out, title="Loop")
        assert Path(result).exists()

    def test_insufficient_data(self, tmp_path):
        out = tmp_path / "short.png"
        result = plot_trajectory([(0, 0)], out)
        assert Path(result).exists()


class TestPlotEdgeCaseSummary:
    def test_creates_png(self, tmp_path):
        edge_cases = [
            _make_edge_case(severity="critical", category="localization"),
            _make_edge_case(severity="warning", category="perception"),
            _make_edge_case(severity="info", category="both"),
        ]
        out = tmp_path / "edge.png"
        result = plot_edge_case_summary(edge_cases, out)
        assert Path(result).exists()
        assert Path(result).stat().st_size > 0

    def test_empty_edge_cases(self, tmp_path):
        out = tmp_path / "edge_empty.png"
        result = plot_edge_case_summary([], out)
        assert Path(result).exists()
