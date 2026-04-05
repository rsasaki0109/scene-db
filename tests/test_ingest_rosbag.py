"""Tests for scene_db.ingest_rosbag."""

import math
from pathlib import Path

import pytest

from scene_db.ingest_rosbag import (
    RosbagFormat,
    detect_rosbag_format,
    _quaternion_to_yaw,
)


class TestRosbagFormatEnum:
    def test_ros1_value(self):
        assert RosbagFormat.ROS1.name == "ROS1"

    def test_ros2_value(self):
        assert RosbagFormat.ROS2.name == "ROS2"

    def test_enum_members(self):
        assert len(RosbagFormat) == 2


class TestDetectRosbagFormat:
    def test_ros1_bag_file(self, tmp_path):
        bag_file = tmp_path / "test.bag"
        bag_file.write_bytes(b"\x00" * 10)  # minimal file
        assert detect_rosbag_format(bag_file) == RosbagFormat.ROS1

    def test_ros2_directory(self, tmp_path):
        bag_dir = tmp_path / "test_bag"
        bag_dir.mkdir()
        (bag_dir / "metadata.yaml").write_text("rosbag2_bagfile_information:\n")
        assert detect_rosbag_format(bag_dir) == RosbagFormat.ROS2

    def test_invalid_file_extension(self, tmp_path):
        txt_file = tmp_path / "data.txt"
        txt_file.write_text("not a bag")
        with pytest.raises(ValueError, match="Cannot detect rosbag format"):
            detect_rosbag_format(txt_file)

    def test_directory_without_metadata(self, tmp_path):
        bag_dir = tmp_path / "no_meta"
        bag_dir.mkdir()
        with pytest.raises(ValueError, match="Cannot detect rosbag format"):
            detect_rosbag_format(bag_dir)

    def test_nonexistent_path(self, tmp_path):
        missing = tmp_path / "nonexistent.bag"
        with pytest.raises(ValueError, match="Cannot detect rosbag format"):
            detect_rosbag_format(missing)


class TestQuaternionToYaw:
    def test_identity_quaternion(self):
        # Identity quaternion (x=0, y=0, z=0, w=1) -> yaw = 0
        yaw = _quaternion_to_yaw(0.0, 0.0, 0.0, 1.0)
        assert yaw == pytest.approx(0.0)

    def test_90_degrees(self):
        # 90 degrees yaw: quaternion (0, 0, sin(45deg), cos(45deg))
        angle = math.pi / 2
        z = math.sin(angle / 2)
        w = math.cos(angle / 2)
        yaw = _quaternion_to_yaw(0.0, 0.0, z, w)
        assert yaw == pytest.approx(math.pi / 2, abs=1e-6)

    def test_180_degrees(self):
        # 180 degrees yaw: quaternion (0, 0, sin(90deg), cos(90deg)) = (0,0,1,0)
        yaw = _quaternion_to_yaw(0.0, 0.0, 1.0, 0.0)
        assert yaw == pytest.approx(math.pi, abs=1e-6)

    def test_negative_90_degrees(self):
        angle = -math.pi / 2
        z = math.sin(angle / 2)
        w = math.cos(angle / 2)
        yaw = _quaternion_to_yaw(0.0, 0.0, z, w)
        assert yaw == pytest.approx(-math.pi / 2, abs=1e-6)

    def test_45_degrees(self):
        angle = math.pi / 4
        z = math.sin(angle / 2)
        w = math.cos(angle / 2)
        yaw = _quaternion_to_yaw(0.0, 0.0, z, w)
        assert yaw == pytest.approx(math.pi / 4, abs=1e-6)
