"""CLI entry point for scene-db."""

from pathlib import Path
from typing import Optional

import typer

from scene_db.db import get_connection, get_scene_by_id, list_all_scenes
from scene_db.export import export_scene
from scene_db.ingest import ingest_sequence
from scene_db.search import search

app = typer.Typer(help="Search and extract scenes from autonomous driving log data.")


@app.command()
def ingest(
    dataset_path: Path = typer.Argument(..., help="Path to dataset directory or bag file"),
    dataset_name: str = typer.Option("auto", help="Format: auto, kitti, nuscenes, rosbag"),
    chunk_duration: float = typer.Option(5.0, help="Chunk duration in seconds"),
    vlm: bool = typer.Option(False, "--vlm", help="Use VLM for richer captions (requires OPENAI_API_KEY)"),
    nuscenes_version: str = typer.Option("v1.0-mini", help="nuScenes version subdirectory"),
    imu_topic: Optional[str] = typer.Option(None, "--imu-topic", help="IMU topic name (rosbag)"),
    odom_topic: Optional[str] = typer.Option(None, "--odom-topic", help="Odometry topic name (rosbag)"),
    db: Optional[Path] = typer.Option(None, help="Database path (default: ~/.scene-db/scene.db)"),
) -> None:
    """Ingest a dataset sequence into the scene database."""
    if not dataset_path.exists():
        typer.echo(f"Error: path does not exist: {dataset_path}", err=True)
        raise typer.Exit(1)

    # Auto-detect format
    fmt = dataset_name
    if fmt == "auto":
        if dataset_path.suffix == ".bag":
            fmt = "rosbag"
        elif (dataset_path / "oxts").exists():
            fmt = "kitti"
        elif any((dataset_path / v).exists() for v in ["v1.0-mini", "v1.0-trainval"]):
            fmt = "nuscenes"
        elif any((dataset_path / c / "run1" / "reference.csv").exists()
                 for c in ["tokyo", "nagoya", "osaka"] if (dataset_path / c).exists()):
            fmt = "ppc"
        else:
            fmt = "kitti"

    typer.echo(f"Ingesting {dataset_path} (format: {fmt}) ...")
    try:
        if fmt == "rosbag":
            from scene_db.ingest_rosbag import ingest_rosbag
            n = ingest_rosbag(
                dataset_path, dataset_name=dataset_path.stem,
                chunk_duration_sec=chunk_duration,
                imu_topic=imu_topic, odom_topic=odom_topic,
                db_path=db,
            )
        elif fmt == "nuscenes":
            from scene_db.ingest_nuscenes import ingest_nuscenes
            n = ingest_nuscenes(dataset_path, nuscenes_version, chunk_duration, db)
        elif fmt == "ppc":
            from scene_db.ingest_ppc import ingest_ppc
            n = ingest_ppc(dataset_path, chunk_duration, db)
        else:
            n = ingest_sequence(dataset_path, fmt, chunk_duration, db, use_vlm=vlm)
    except FileNotFoundError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
    typer.echo(f"Done. Created {n} scene chunks.")


@app.command()
def index(
    embed: bool = typer.Option(False, "--embed", help="Build embedding index for semantic search"),
    db: Optional[Path] = typer.Option(None, help="Database path"),
) -> None:
    """Show index status or build embedding index."""
    conn = get_connection(db)
    scenes = list_all_scenes(conn)
    typer.echo(f"Index contains {len(scenes)} scenes.")

    if embed:
        try:
            from scene_db.embedding import build_embeddings
            n = build_embeddings(conn)
            typer.echo(f"Built embeddings for {n} scenes.")
        except RuntimeError as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(1)

    conn.close()


@app.command(name="search")
def search_cmd(
    query: str = typer.Argument("", help="Search query text (empty = all scenes)"),
    semantic: bool = typer.Option(False, "--semantic", "-s", help="Use semantic search (requires embeddings)"),
    top_k: int = typer.Option(10, "-k", help="Number of results for semantic search"),
    min_speed: Optional[float] = typer.Option(None, "--min-speed", help="Minimum avg speed (km/h)"),
    max_speed: Optional[float] = typer.Option(None, "--max-speed", help="Maximum avg speed (km/h)"),
    min_decel: Optional[float] = typer.Option(None, "--min-decel", help="Minimum max deceleration (m/s^2)"),
    min_yaw: Optional[float] = typer.Option(None, "--min-yaw", help="Minimum max yaw rate (deg/s)"),
    min_accel: Optional[float] = typer.Option(None, "--min-accel", help="Minimum max acceleration (m/s^2)"),
    sort: Optional[str] = typer.Option(None, "--sort", help="Sort by: speed, decel, yaw, accel"),
    db: Optional[Path] = typer.Option(None, help="Database path"),
) -> None:
    """Search scenes by text query and/or feature filters."""
    conn = get_connection(db)

    if semantic:
        try:
            from scene_db.embedding import semantic_search
            results = semantic_search(conn, query, top_k)
        except RuntimeError as e:
            typer.echo(f"Error: {e}", err=True)
            conn.close()
            raise typer.Exit(1)

        if not results:
            typer.echo("No scenes found.")
            conn.close()
            return

        typer.echo(f"Found {len(results)} scene(s):\n")
        for scene_id, score in results:
            chunk = get_scene_by_id(conn, scene_id)
            if chunk:
                typer.echo(f"  [{chunk.id}] (score: {score:.3f})")
                typer.echo(f"    {chunk.caption}")
                typer.echo(f"    frames {chunk.start_frame}-{chunk.end_frame}, "
                           f"{chunk.start_time.isoformat()} - {chunk.end_time.isoformat()}")
                typer.echo()
    else:
        results = search(
            conn, query,
            min_speed=min_speed, max_speed=max_speed,
            min_decel=min_decel, min_yaw=min_yaw, min_accel=min_accel,
            sort_by=sort,
        )
        if not results:
            typer.echo("No scenes found.")
            conn.close()
            return

        typer.echo(f"Found {len(results)} scene(s):\n")
        for s in results:
            typer.echo(f"  [{s.id}]")
            typer.echo(f"    {s.caption}")
            typer.echo(f"    frames {s.start_frame}-{s.end_frame}, "
                       f"{s.start_time.isoformat()} - {s.end_time.isoformat()}")
            details = []
            if s.max_decel_ms2 > 0.5:
                details.append(f"decel {s.max_decel_ms2:.1f} m/s\u00b2")
            if s.max_yaw_rate_degs > 1.0:
                details.append(f"yaw {s.max_yaw_rate_degs:.1f} \u00b0/s")
            if details:
                typer.echo(f"    [{', '.join(details)}]")
            typer.echo()

    conn.close()


@app.command(name="edge-cases")
def edge_cases_cmd(
    category: Optional[str] = typer.Option(None, "--category", "-c", help="Filter: localization, perception, both"),
    severity: Optional[str] = typer.Option(None, "--severity", help="Filter: critical, warning, info"),
    limit: int = typer.Option(20, "-n", help="Max results to show"),
    db: Optional[Path] = typer.Option(None, help="Database path"),
) -> None:
    """Detect and list edge cases for localization and perception."""
    from scene_db.edge_detect import detect_edge_cases

    conn = get_connection(db)
    cases = detect_edge_cases(conn)
    conn.close()

    if category:
        cases = [c for c in cases if c.category == category]
    if severity:
        cases = [c for c in cases if c.severity == severity]

    if not cases:
        typer.echo("No edge cases detected.")
        return

    # Summary
    n_crit = sum(1 for c in cases if c.severity == "critical")
    n_warn = sum(1 for c in cases if c.severity == "warning")
    n_info = sum(1 for c in cases if c.severity == "info")
    typer.echo(f"Detected {len(cases)} edge cases: "
               f"{n_crit} critical, {n_warn} warning, {n_info} info\n")

    severity_icons = {"critical": "\u2716", "warning": "\u26a0", "info": "\u2139"}
    category_colors = {"localization": "LOC", "perception": "PER", "both": "L+P"}

    for c in cases[:limit]:
        icon = severity_icons[c.severity]
        cat = category_colors.get(c.category, c.category)
        typer.echo(f"  {icon} [{cat}] [{c.scene.id}]")
        typer.echo(f"    {c.reason}")
        typer.echo(f"    {c.scene.caption}")
        typer.echo(f"    score: {c.score:.2f} | frames {c.scene.start_frame}-{c.scene.end_frame}")
        typer.echo()


@app.command()
def stats(
    db: Optional[Path] = typer.Option(None, help="Database path"),
) -> None:
    """Show statistics about the scene database."""
    conn = get_connection(db)
    scenes = list_all_scenes(conn)
    conn.close()

    if not scenes:
        typer.echo("Database is empty.")
        return

    # Aggregate stats
    sequences = set()
    datasets = set()
    total_frames = 0
    speeds = []
    decels = []
    yaw_rates = []

    for s in scenes:
        sequences.add(f"{s.dataset_name}/{s.sequence_id}")
        datasets.add(s.dataset_name)
        total_frames += s.end_frame - s.start_frame + 1
        speeds.append(s.avg_speed_kmh)
        decels.append(s.max_decel_ms2)
        yaw_rates.append(s.max_yaw_rate_degs)

    # Caption category counts
    categories: dict[str, int] = {}
    for s in scenes:
        for kw in ["stationary", "moving slowly", "moving forward", "high speed",
                    "braking", "hard braking", "turning", "sharp turn", "gentle curve"]:
            if kw in s.caption:
                categories[kw] = categories.get(kw, 0) + 1

    typer.echo(f"Scenes: {len(scenes)}  |  Sequences: {len(sequences)}  |  Datasets: {', '.join(datasets)}")
    typer.echo(f"Total frames: {total_frames}")
    typer.echo()
    typer.echo("Speed distribution:")
    typer.echo(f"  min {min(speeds):.0f} / avg {sum(speeds)/len(speeds):.0f} / max {max(speeds):.0f} km/h")
    typer.echo()
    typer.echo("Max deceleration:")
    typer.echo(f"  min {min(decels):.1f} / avg {sum(decels)/len(decels):.1f} / max {max(decels):.1f} m/s\u00b2")
    typer.echo()
    typer.echo("Max yaw rate:")
    typer.echo(f"  min {min(yaw_rates):.1f} / avg {sum(yaw_rates)/len(yaw_rates):.1f} / max {max(yaw_rates):.1f} \u00b0/s")
    typer.echo()
    typer.echo("Scene categories:")
    for kw, count in sorted(categories.items(), key=lambda x: -x[1]):
        bar = "\u2588" * count
        typer.echo(f"  {kw:16s} {count:3d} {bar}")


@app.command()
def export(
    id: str = typer.Option(..., "--id", help="Scene chunk ID to export"),
    output: Path = typer.Option("./export", "-o", "--output", help="Output directory"),
    db: Optional[Path] = typer.Option(None, help="Database path"),
) -> None:
    """Export a scene's files to a directory."""
    conn = get_connection(db)
    try:
        n = export_scene(conn, id, output)
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
    finally:
        conn.close()
    typer.echo(f"Exported {n} files to {output}")
