"""Sequence-level analysis: total distance, loop detection, revisit detection."""

import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime

from scene_db.models import SceneChunk


@dataclass
class SequenceInfo:
    """Aggregated info for an entire sequence."""

    dataset_name: str
    sequence_id: str
    total_chunks: int
    total_frames: int
    total_distance_m: float
    duration_sec: float
    avg_speed_kmh: float
    max_speed_kmh: float
    max_decel_ms2: float
    max_yaw_rate_degs: float
    has_loop: bool
    loop_distance_m: float  # distance between start and end positions
    revisit_count: int  # number of times the trajectory revisits a previous location
    start_time: datetime
    end_time: datetime


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute distance in meters between two lat/lon points."""
    R = 6371000.0  # Earth radius in meters
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _euclidean_2d(x1: float, y1: float, x2: float, y2: float) -> float:
    """2D Euclidean distance."""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def _detect_loop_from_positions(
    positions: list[tuple[float, float]],
    threshold_m: float = 10.0,
) -> tuple[bool, float, int]:
    """Detect if trajectory forms a loop and count revisits.

    Args:
        positions: list of (x, y) positions in local frame
        threshold_m: distance threshold for revisit detection

    Returns:
        (has_loop, loop_distance, revisit_count)
    """
    if len(positions) < 2:
        return False, 0.0, 0

    start = positions[0]
    end = positions[-1]
    loop_distance = _euclidean_2d(start[0], start[1], end[0], end[1])
    has_loop = loop_distance < threshold_m

    # Count revisits: check if any later position comes close to an earlier one
    # Use spatial grid for efficiency
    grid_size = threshold_m
    grid: dict[tuple[int, int], int] = {}
    revisit_count = 0
    min_gap = max(10, len(positions) // 20)  # minimum index gap to count as revisit

    for i, (x, y) in enumerate(positions):
        cell = (int(x / grid_size), int(y / grid_size))
        # Check neighboring cells
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                neighbor = (cell[0] + dx, cell[1] + dy)
                if neighbor in grid and (i - grid[neighbor]) > min_gap:
                    revisit_count += 1
                    break
            else:
                continue
            break
        grid[cell] = i

    return has_loop, loop_distance, revisit_count


def _lat_lon_to_local(
    lat: float, lon: float, ref_lat: float, ref_lon: float,
) -> tuple[float, float]:
    """Convert lat/lon to local XY (meters) relative to reference."""
    R = 6371000.0
    x = R * math.radians(lon - ref_lon) * math.cos(math.radians(ref_lat))
    y = R * math.radians(lat - ref_lat)
    return x, y


def analyze_sequences(conn: sqlite3.Connection, loop_threshold_m: float = 10.0) -> list[SequenceInfo]:
    """Analyze all sequences in the database for distance, loops, and revisits."""
    # Get all chunks grouped by dataset+sequence
    cursor = conn.execute(
        """SELECT id, dataset_name, sequence_id, chunk_index,
                  start_time, end_time, start_frame, end_frame,
                  avg_speed_kmh, distance_m,
                  max_accel_ms2, max_decel_ms2,
                  avg_yaw_rate_degs, max_yaw_rate_degs,
                  caption
           FROM scene_chunks
           ORDER BY dataset_name, sequence_id, chunk_index"""
    )

    from scene_db.db import _row_to_chunk

    # Group chunks by sequence
    sequences: dict[str, list[SceneChunk]] = {}
    for row in cursor.fetchall():
        chunk = _row_to_chunk(row)
        key = f"{chunk.dataset_name}/{chunk.sequence_id}"
        sequences.setdefault(key, []).append(chunk)

    # Try to get position data from file_refs if available
    # For KITTI/PPC we have lat/lon in the original data
    # For rosbag we have positions from odometry

    results = []
    for key, chunks in sequences.items():
        dataset_name = chunks[0].dataset_name
        sequence_id = chunks[0].sequence_id

        total_distance = sum(c.distance_m for c in chunks)
        total_frames = sum(c.end_frame - c.start_frame + 1 for c in chunks)
        duration = (chunks[-1].end_time - chunks[0].start_time).total_seconds()
        speeds = [c.avg_speed_kmh for c in chunks]
        avg_speed = sum(speeds) / len(speeds) if speeds else 0
        max_speed = max(speeds) if speeds else 0
        max_decel = max(c.max_decel_ms2 for c in chunks)
        max_yaw = max(c.max_yaw_rate_degs for c in chunks)

        # Try to detect loops using lat/lon if available
        # We check if chunks have meaningful lat/lon (KITTI, PPC have this)
        positions = _get_sequence_positions(conn, dataset_name, sequence_id)
        if positions:
            has_loop, loop_dist, revisit_count = _detect_loop_from_positions(
                positions, loop_threshold_m
            )
        else:
            # Estimate from cumulative distance vs displacement
            # If total distance >> 0 but we can't measure displacement, mark unknown
            has_loop = False
            loop_dist = -1.0  # unknown
            revisit_count = 0

        results.append(SequenceInfo(
            dataset_name=dataset_name,
            sequence_id=sequence_id,
            total_chunks=len(chunks),
            total_frames=total_frames,
            total_distance_m=total_distance,
            duration_sec=duration,
            avg_speed_kmh=avg_speed,
            max_speed_kmh=max_speed,
            max_decel_ms2=max_decel,
            max_yaw_rate_degs=max_yaw,
            has_loop=has_loop,
            loop_distance_m=loop_dist,
            revisit_count=revisit_count,
            start_time=chunks[0].start_time,
            end_time=chunks[-1].end_time,
        ))

    results.sort(key=lambda s: s.total_distance_m, reverse=True)
    return results


def _get_sequence_positions(
    conn: sqlite3.Connection, dataset_name: str, sequence_id: str,
) -> list[tuple[float, float]]:
    """Try to reconstruct trajectory positions for a sequence.

    For KITTI/PPC: read from the original data files (lat/lon).
    For rosbag: use lat/lon stored in OxtsRecord (which may be odom x,y).
    """
    # Get the first file ref to determine data source
    cursor = conn.execute(
        """SELECT sc.id, fr.file_type, fr.file_path
           FROM scene_chunks sc
           JOIN file_refs fr ON fr.scene_id = sc.id
           WHERE sc.dataset_name = ? AND sc.sequence_id = ?
           ORDER BY sc.chunk_index, fr.frame_index
           LIMIT 1""",
        (dataset_name, sequence_id),
    )
    row = cursor.fetchone()
    if row is None:
        return []

    file_type = row[1]
    file_path = row[2]

    if file_type == "oxts":
        return _read_kitti_positions(conn, dataset_name, sequence_id)
    elif file_type == "reference":
        return _read_ppc_positions(file_path)
    elif file_type == "rosbag":
        return _read_rosbag_positions(conn, dataset_name, sequence_id)

    return []


def _read_kitti_positions(
    conn: sqlite3.Connection, dataset_name: str, sequence_id: str,
) -> list[tuple[float, float]]:
    """Read positions from KITTI oxts files."""
    import pathlib

    cursor = conn.execute(
        """SELECT fr.file_path
           FROM scene_chunks sc
           JOIN file_refs fr ON fr.scene_id = sc.id
           WHERE sc.dataset_name = ? AND sc.sequence_id = ?
             AND fr.file_type = 'oxts'
           ORDER BY sc.chunk_index, fr.frame_index""",
        (dataset_name, sequence_id),
    )
    rows = cursor.fetchall()
    if not rows:
        return []

    positions = []
    ref_lat = ref_lon = None
    seen = set()

    for (file_path,) in rows:
        if file_path in seen:
            continue
        seen.add(file_path)
        p = pathlib.Path(file_path)
        if not p.exists():
            continue
        values = p.read_text().strip().split()
        if len(values) < 2:
            continue
        lat, lon = float(values[0]), float(values[1])
        if ref_lat is None:
            ref_lat, ref_lon = lat, lon
        x, y = _lat_lon_to_local(lat, lon, ref_lat, ref_lon)
        positions.append((x, y))

    return positions


def _read_ppc_positions(reference_csv_path: str) -> list[tuple[float, float]]:
    """Read positions from PPC reference.csv."""
    import csv
    import pathlib

    p = pathlib.Path(reference_csv_path)
    if not p.exists():
        return []

    positions = []
    ref_lat = ref_lon = None

    with open(p) as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if len(row) < 4:
                continue
            lat, lon = float(row[2]), float(row[3])
            if ref_lat is None:
                ref_lat, ref_lon = lat, lon
            x, y = _lat_lon_to_local(lat, lon, ref_lat, ref_lon)
            positions.append((x, y))

    return positions


def _read_rosbag_positions(
    conn: sqlite3.Connection, dataset_name: str, sequence_id: str,
) -> list[tuple[float, float]]:
    """For rosbag data, lat/lon fields store odom x,y positions."""
    cursor = conn.execute(
        """SELECT sc.id, sc.chunk_index
           FROM scene_chunks sc
           WHERE sc.dataset_name = ? AND sc.sequence_id = ?
           ORDER BY sc.chunk_index""",
        (dataset_name, sequence_id),
    )
    # Rosbag positions were stored in lat/lon fields of OxtsRecord
    # but we don't persist individual records to DB.
    # Return empty - loop detection requires re-reading the bag.
    return []
