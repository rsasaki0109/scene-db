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
    dataset_path: Path = typer.Argument(..., help="Path to dataset directory"),
    dataset_name: str = typer.Option("kitti", help="Dataset name (kitti or nuscenes)"),
    chunk_duration: float = typer.Option(5.0, help="Chunk duration in seconds"),
    vlm: bool = typer.Option(False, "--vlm", help="Use VLM for richer captions (requires OPENAI_API_KEY)"),
    nuscenes_version: str = typer.Option("v1.0-mini", help="nuScenes version subdirectory"),
    db: Optional[Path] = typer.Option(None, help="Database path (default: ~/.scene-db/scene.db)"),
) -> None:
    """Ingest a dataset sequence into the scene database."""
    if not dataset_path.exists():
        typer.echo(f"Error: path does not exist: {dataset_path}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Ingesting {dataset_path} (format: {dataset_name}) ...")
    try:
        if dataset_name == "nuscenes":
            from scene_db.ingest_nuscenes import ingest_nuscenes
            n = ingest_nuscenes(dataset_path, nuscenes_version, chunk_duration, db)
        else:
            n = ingest_sequence(dataset_path, dataset_name, chunk_duration, db, use_vlm=vlm)
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
    query: str = typer.Argument(..., help="Search query text"),
    semantic: bool = typer.Option(False, "--semantic", "-s", help="Use semantic search (requires embeddings)"),
    top_k: int = typer.Option(10, "-k", help="Number of results for semantic search"),
    db: Optional[Path] = typer.Option(None, help="Database path"),
) -> None:
    """Search scenes by text query."""
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
        results = search(conn, query)
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
