"""Rosbag (ROS1) ingestion for LiDAR SLAM datasets."""

import math
import struct
from datetime import datetime
from pathlib import Path

from scene_db.db import get_connection, insert_scene_chunks
from scene_db.features import extract_features, generate_caption
from scene_db.models import FileRef, OxtsRecord, SceneChunk


def _parse_imu_msg(rawdata: bytes, typestore) -> dict | None:
    """Parse sensor_msgs/Imu message."""
    try:
        msg = typestore.deserialize_ros1(rawdata, "sensor_msgs/msg/Imu")
        return {
            "orientation": (
                msg.orientation.x,
                msg.orientation.y,
                msg.orientation.z,
                msg.orientation.w,
            ),
            "angular_velocity": (
                msg.angular_velocity.x,
                msg.angular_velocity.y,
                msg.angular_velocity.z,
            ),
            "linear_acceleration": (
                msg.linear_acceleration.x,
                msg.linear_acceleration.y,
                msg.linear_acceleration.z,
            ),
        }
    except Exception:
        return None


def _quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    """Extract yaw from quaternion."""
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def _parse_odom_msg(rawdata: bytes, typestore) -> dict | None:
    """Parse nav_msgs/Odometry message."""
    try:
        msg = typestore.deserialize_ros1(rawdata, "nav_msgs/msg/Odometry")
        return {
            "position": (
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z,
            ),
            "orientation": (
                msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z,
                msg.pose.pose.orientation.w,
            ),
            "linear_velocity": (
                msg.twist.twist.linear.x,
                msg.twist.twist.linear.y,
                msg.twist.twist.linear.z,
            ),
        }
    except Exception:
        return None


def _detect_topics(reader) -> dict[str, str]:
    """Auto-detect IMU and point cloud topics from a bag."""
    topics: dict[str, str] = {}
    for conn in reader.connections:
        topic = conn.topic
        msgtype = conn.msgtype
        if "Imu" in msgtype and "imu" not in topics:
            topics["imu"] = topic
        elif "PointCloud2" in msgtype and "points" not in topics:
            topics["points"] = topic
        elif "Odometry" in msgtype and "odom" not in topics:
            topics["odom"] = topic
    return topics


def _read_imu_records(
    bag_path: Path,
    imu_topic: str | None = None,
    odom_topic: str | None = None,
) -> list[OxtsRecord]:
    """Read IMU and/or odometry data from a rosbag, return as OxtsRecord list."""
    from rosbags.rosbag1 import Reader
    from rosbags.typesys import Stores, get_typestore

    typestore = get_typestore(Stores.ROS1_NOETIC)
    records = []

    with Reader(bag_path) as reader:
        detected = _detect_topics(reader)
        imu_topic = imu_topic or detected.get("imu")
        odom_topic = odom_topic or detected.get("odom")

        # Prefer odometry (has velocity), fall back to IMU
        if odom_topic:
            target_topic = odom_topic
            connections = [c for c in reader.connections if c.topic == target_topic]
            if connections:
                for i, (conn, timestamp, rawdata) in enumerate(
                    reader.messages(connections=connections)
                ):
                    odom = _parse_odom_msg(rawdata, typestore)
                    if odom is None:
                        continue
                    ts = datetime.fromtimestamp(timestamp / 1e9)
                    vx, vy, vz = odom["linear_velocity"]
                    ox, oy, oz, ow = odom["orientation"]
                    yaw = _quaternion_to_yaw(ox, oy, oz, ow)
                    records.append(
                        OxtsRecord(
                            timestamp=ts,
                            frame_index=i,
                            lat=odom["position"][0],
                            lon=odom["position"][1],
                            alt=odom["position"][2],
                            roll=0.0,
                            pitch=0.0,
                            yaw=yaw,
                            vf=vx,
                            vl=vy,
                            vu=vz,
                        )
                    )

        if not records and imu_topic:
            target_topic = imu_topic
            connections = [c for c in reader.connections if c.topic == target_topic]
            prev_yaw = None
            prev_ts = None
            vf = 0.0  # Integrate acceleration for velocity estimate
            if connections:
                for i, (conn, timestamp, rawdata) in enumerate(
                    reader.messages(connections=connections)
                ):
                    imu = _parse_imu_msg(rawdata, typestore)
                    if imu is None:
                        continue
                    ts = datetime.fromtimestamp(timestamp / 1e9)
                    ox, oy, oz, ow = imu["orientation"]
                    yaw = _quaternion_to_yaw(ox, oy, oz, ow)
                    ax, ay, az = imu["linear_acceleration"]

                    # Rough velocity estimate from acceleration integration
                    if prev_ts is not None:
                        dt = (ts - prev_ts).total_seconds()
                        if 0 < dt < 1.0:
                            vf += ax * dt

                    # Subsample IMU (typically 100-400 Hz, we want ~10 Hz)
                    if i % 10 == 0:
                        records.append(
                            OxtsRecord(
                                timestamp=ts,
                                frame_index=len(records),
                                lat=0.0,
                                lon=0.0,
                                alt=0.0,
                                roll=0.0,
                                pitch=0.0,
                                yaw=yaw,
                                vf=vf,
                                vl=0.0,
                                vu=0.0,
                            )
                        )
                    prev_yaw = yaw
                    prev_ts = ts

    return records


def _count_pointcloud_frames(bag_path: Path, points_topic: str | None = None) -> int:
    """Count the number of point cloud frames in a bag."""
    from rosbags.rosbag1 import Reader

    with Reader(bag_path) as reader:
        if points_topic is None:
            detected = _detect_topics(reader)
            points_topic = detected.get("points")
        if points_topic is None:
            return 0
        connections = [c for c in reader.connections if c.topic == points_topic]
        return sum(1 for _ in reader.messages(connections=connections))


def ingest_rosbag(
    bag_path: Path,
    dataset_name: str = "rosbag",
    chunk_duration_sec: float = 5.0,
    imu_topic: str | None = None,
    odom_topic: str | None = None,
    points_topic: str | None = None,
    db_path: Path | None = None,
) -> int:
    """Ingest a ROS1 bag file into the scene database. Returns number of chunks created."""
    if not bag_path.exists():
        raise FileNotFoundError(f"Bag file not found: {bag_path}")

    records = _read_imu_records(bag_path, imu_topic, odom_topic)
    if not records:
        return 0

    # Split into chunks
    from scene_db.ingest import split_into_chunks

    chunk_ranges = split_into_chunks(records, chunk_duration_sec)
    sequence_id = bag_path.stem

    scene_chunks = []
    for chunk_idx, (start_idx, end_idx) in enumerate(chunk_ranges):
        chunk_records = records[start_idx : end_idx + 1]
        feat = extract_features(chunk_records)
        caption = generate_caption(
            feat.avg_speed_kmh,
            feat.distance_m,
            feat.max_decel_ms2,
            feat.avg_yaw_rate_degs,
            feat.max_yaw_rate_degs,
        )

        chunk_id = f"{dataset_name}_{sequence_id}_{chunk_idx:03d}"

        # Bag file itself as a file ref
        file_refs = [
            FileRef(
                scene_id=chunk_id,
                file_type="rosbag",
                frame_index=start_idx,
                file_path=str(bag_path),
            )
        ]

        scene_chunks.append(
            SceneChunk(
                id=chunk_id,
                dataset_name=dataset_name,
                sequence_id=sequence_id,
                chunk_index=chunk_idx,
                start_time=chunk_records[0].timestamp,
                end_time=chunk_records[-1].timestamp,
                start_frame=start_idx,
                end_frame=end_idx,
                avg_speed_kmh=feat.avg_speed_kmh,
                distance_m=feat.distance_m,
                max_accel_ms2=feat.max_accel_ms2,
                max_decel_ms2=feat.max_decel_ms2,
                avg_yaw_rate_degs=feat.avg_yaw_rate_degs,
                max_yaw_rate_degs=feat.max_yaw_rate_degs,
                caption=caption,
                file_refs=file_refs,
            )
        )

    conn = get_connection(db_path)
    try:
        insert_scene_chunks(conn, scene_chunks)
    finally:
        conn.close()

    return len(scene_chunks)
