"""nuScenes dataset ingestion."""

import json
from datetime import datetime
from pathlib import Path

from scene_db.db import get_connection, insert_scene_chunks
from scene_db.features import compute_avg_speed_kmh, compute_distance_m, generate_caption
from scene_db.models import FileRef, OxtsRecord, SceneChunk


def _load_table(dataroot: Path, table_name: str) -> list[dict]:
    """Load a nuScenes table JSON file."""
    path = dataroot / table_name
    if not path.exists():
        path = dataroot / f"{table_name}.json"
    if not path.exists():
        raise FileNotFoundError(f"nuScenes table not found: {path}")
    return json.loads(path.read_text())


def _build_token_map(records: list[dict]) -> dict[str, dict]:
    """Build a lookup dict from token to record."""
    return {r["token"]: r for r in records}


def _ego_pose_to_speed(
    ego_poses: list[dict], timestamps_us: list[int]
) -> list[float]:
    """Estimate speed from consecutive ego poses (m/s)."""
    speeds = [0.0]
    for i in range(1, len(ego_poses)):
        dt = (timestamps_us[i] - timestamps_us[i - 1]) / 1e6
        if dt <= 0:
            speeds.append(0.0)
            continue
        dx = ego_poses[i]["translation"][0] - ego_poses[i - 1]["translation"][0]
        dy = ego_poses[i]["translation"][1] - ego_poses[i - 1]["translation"][1]
        speed = (dx**2 + dy**2) ** 0.5 / dt
        speeds.append(speed)
    return speeds


def ingest_nuscenes(
    dataroot: Path,
    version: str = "v1.0-mini",
    chunk_duration_sec: float = 5.0,
    db_path: Path | None = None,
) -> int:
    """Ingest nuScenes dataset. Returns number of chunks created."""
    meta_dir = dataroot / version

    # Load metadata tables
    scenes = _load_table(meta_dir, "scene")
    samples = _load_table(meta_dir, "sample")
    sample_data_list = _load_table(meta_dir, "sample_data")
    ego_poses = _load_table(meta_dir, "ego_pose")

    sample_map = _build_token_map(samples)
    sample_data_map = _build_token_map(sample_data_list)
    ego_pose_map = _build_token_map(ego_poses)

    all_chunks = []

    for scene in scenes:
        scene_name = scene["name"]
        scene_token = scene["token"]

        # Collect all samples in this scene
        scene_samples = []
        sample_token = scene["first_sample_token"]
        while sample_token:
            sample = sample_map.get(sample_token)
            if sample is None:
                break
            scene_samples.append(sample)
            sample_token = sample.get("next", "")

        if not scene_samples:
            continue

        # Get timestamps and ego poses for each sample
        timestamps_us = [s["timestamp"] for s in scene_samples]
        sample_ego_poses = []
        for s in scene_samples:
            # Get ego_pose from the first sample_data (CAM_FRONT typically)
            sd_token = s["data"].get("CAM_FRONT") if isinstance(s.get("data"), dict) else None
            if sd_token and sd_token in sample_data_map:
                ep_token = sample_data_map[sd_token].get("ego_pose_token", "")
                ep = ego_pose_map.get(ep_token)
                if ep:
                    sample_ego_poses.append(ep)
                    continue
            sample_ego_poses.append({"translation": [0, 0, 0]})

        # Convert to OxtsRecord-like for feature computation
        speeds = _ego_pose_to_speed(sample_ego_poses, timestamps_us)
        records = []
        for i, s in enumerate(scene_samples):
            ts = datetime.fromtimestamp(timestamps_us[i] / 1e6)
            records.append(
                OxtsRecord(
                    timestamp=ts,
                    frame_index=i,
                    lat=0.0, lon=0.0, alt=0.0,
                    roll=0.0, pitch=0.0, yaw=0.0,
                    vf=speeds[i], vl=0.0, vu=0.0,
                )
            )

        # Split into chunks
        chunk_start = 0
        chunk_ranges = []
        for i in range(1, len(records)):
            elapsed = (records[i].timestamp - records[chunk_start].timestamp).total_seconds()
            if elapsed >= chunk_duration_sec:
                chunk_ranges.append((chunk_start, i - 1))
                chunk_start = i
        if chunk_start < len(records):
            chunk_ranges.append((chunk_start, len(records) - 1))

        # Build scene chunks
        for chunk_idx, (start_idx, end_idx) in enumerate(chunk_ranges):
            chunk_records = records[start_idx : end_idx + 1]
            avg_speed = compute_avg_speed_kmh(chunk_records)
            distance = compute_distance_m(chunk_records)
            caption = generate_caption(avg_speed, distance)

            chunk_id = f"nuscenes_{scene_name}_{chunk_idx:03d}"

            # Collect file refs from sample_data
            file_refs = []
            for fi in range(start_idx, end_idx + 1):
                sample = scene_samples[fi]
                # sample["data"] is a dict of {channel: sample_data_token}
                data_dict = sample.get("data", {})
                if isinstance(data_dict, dict):
                    for channel, sd_token in data_dict.items():
                        sd = sample_data_map.get(sd_token)
                        if sd and "filename" in sd:
                            file_refs.append(
                                FileRef(
                                    scene_id=chunk_id,
                                    file_type=channel,
                                    frame_index=fi,
                                    file_path=str(dataroot / sd["filename"]),
                                )
                            )

            all_chunks.append(
                SceneChunk(
                    id=chunk_id,
                    dataset_name="nuscenes",
                    sequence_id=scene_name,
                    chunk_index=chunk_idx,
                    start_time=chunk_records[0].timestamp,
                    end_time=chunk_records[-1].timestamp,
                    start_frame=start_idx,
                    end_frame=end_idx,
                    avg_speed_kmh=avg_speed,
                    distance_m=distance,
                    caption=caption,
                    file_refs=file_refs,
                )
            )

    if not all_chunks:
        return 0

    conn = get_connection(db_path)
    try:
        insert_scene_chunks(conn, all_chunks)
    finally:
        conn.close()

    return len(all_chunks)
