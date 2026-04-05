"""PPC (Precise Positioning Challenge) dataset ingestion.

CSV-based GNSS/IMU dataset from urban environments in Japan (Nagoya, Tokyo).
https://github.com/taroz/PPC-Dataset
"""

import csv
import math
from datetime import datetime, timedelta
from pathlib import Path

from scene_db.db import get_connection, insert_scene_chunks
from scene_db.features import extract_features, generate_caption
from scene_db.ingest import split_into_chunks
from scene_db.models import FileRef, OxtsRecord, SceneChunk


def _gps_tow_to_datetime(tow: float, week: int) -> datetime:
    """Convert GPS Time of Week to datetime."""
    gps_epoch = datetime(1980, 1, 6)
    return gps_epoch + timedelta(weeks=week, seconds=tow)


def parse_reference_csv(csv_path: Path) -> list[OxtsRecord]:
    """Parse PPC reference.csv (ground truth with velocity and heading)."""
    records = []
    with open(csv_path) as f:
        reader = csv.reader(f)
        header = next(reader)  # skip header
        for i, row in enumerate(reader):
            if len(row) < 14:
                continue
            tow = float(row[0])
            week = int(row[1])
            lat = float(row[2])
            lon = float(row[3])
            alt = float(row[4])
            roll = math.radians(float(row[8]))
            pitch = math.radians(float(row[9]))
            heading_deg = float(row[10])
            # PPC heading: clockwise from north. Convert to yaw (CCW from east)
            yaw = math.radians(90.0 - heading_deg)
            ve = float(row[11])  # East velocity
            vn = float(row[12])  # North velocity
            vu = float(row[13])  # Up velocity
            # Forward velocity = speed in heading direction
            vf = math.sqrt(ve**2 + vn**2)

            ts = _gps_tow_to_datetime(tow, week)
            records.append(
                OxtsRecord(
                    timestamp=ts,
                    frame_index=i,
                    lat=lat,
                    lon=lon,
                    alt=alt,
                    roll=roll,
                    pitch=pitch,
                    yaw=yaw,
                    vf=vf,
                    vl=0.0,
                    vu=vu,
                )
            )
    return records


def parse_imu_csv(csv_path: Path) -> list[dict]:
    """Parse PPC imu.csv (100 Hz acceleration and angular velocity)."""
    records = []
    with open(csv_path) as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if len(row) < 8:
                continue
            records.append({
                "tow": float(row[0]),
                "week": int(row[1]),
                "ax": float(row[2]),
                "ay": float(row[3]),
                "az": float(row[4]),
                "gx": float(row[5]),
                "gy": float(row[6]),
                "gz": float(row[7]),
            })
    return records


def ingest_ppc_run(
    run_dir: Path,
    dataset_name: str = "ppc",
    chunk_duration_sec: float = 5.0,
    db_path: Path | None = None,
) -> int:
    """Ingest a single PPC run directory. Returns number of chunks created."""
    ref_path = run_dir / "reference.csv"
    imu_path = run_dir / "imu.csv"

    if not ref_path.exists():
        raise FileNotFoundError(f"reference.csv not found: {ref_path}")

    records = parse_reference_csv(ref_path)
    if not records:
        return 0

    # Derive sequence name from path: e.g., "tokyo_run1"
    city = run_dir.parent.name
    run = run_dir.name
    sequence_id = f"{city}_{run}"

    chunk_ranges = split_into_chunks(records, chunk_duration_sec)

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

        file_refs = []
        if ref_path.exists():
            file_refs.append(FileRef(chunk_id, "reference", start_idx, str(ref_path)))
        if imu_path.exists():
            file_refs.append(FileRef(chunk_id, "imu", start_idx, str(imu_path)))

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


def ingest_ppc(
    dataset_dir: Path,
    chunk_duration_sec: float = 5.0,
    db_path: Path | None = None,
) -> int:
    """Ingest all PPC runs from a dataset directory. Returns total chunks created."""
    total = 0
    # Look for city/run directories
    for city_dir in sorted(dataset_dir.iterdir()):
        if not city_dir.is_dir():
            continue
        for run_dir in sorted(city_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            if (run_dir / "reference.csv").exists():
                n = ingest_ppc_run(run_dir, chunk_duration_sec=chunk_duration_sec, db_path=db_path)
                total += n
    return total
