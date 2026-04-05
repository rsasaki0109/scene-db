# scene-db

**Search and extract scenes from autonomous driving and robotics log data.**

scene-db splits driving logs into time-based scene chunks, extracts features, and lets you search and export specific scenes via a simple CLI.

## Features

- **Multi-dataset support**: KITTI raw data and nuScenes
- Time-based scene chunking (configurable duration)
- Automatic feature extraction (speed, distance)
- Rule-based scene captioning + **VLM captioning** (OpenAI GPT-4o)
- Text search + **semantic search** (sentence-transformers / OpenAI embeddings)
- Scene export to directory

## Installation

```bash
pip install -e .                    # Core (KITTI + text search)
pip install -e ".[embedding]"       # + semantic search (sentence-transformers)
pip install -e ".[vlm]"             # + VLM captioning (OpenAI)
pip install -e ".[all]"             # Everything
```

## Quick Start

### 1. Ingest a KITTI sequence

```bash
scene-db ingest /path/to/2011_09_26_drive_0001_sync
```

This parses oxts (GPS/IMU) data, splits the sequence into 5-second chunks, computes speed/distance features, and stores everything in a local SQLite database (`~/.scene-db/scene.db`).

### 2. Search for scenes

```bash
scene-db search "moving forward"
```

```
Found 2 scene(s):

  [kitti_2011_09_26_drive_0001_sync_000]
    vehicle moving forward, 26 km/h, traveled 32.6 m
    frames 0-9, 2011-09-26T13:02:25 - 2011-09-26T13:02:29.500000

  [kitti_2011_09_26_drive_0001_sync_001]
    vehicle moving forward, 44 km/h, traveled 55.1 m
    frames 10-19, 2011-09-26T13:02:30 - 2011-09-26T13:02:34.500000
```

### 3. Export a scene

```bash
scene-db export --id kitti_2011_09_26_drive_0001_sync_000 -o ./my_scene
```

Copies all associated files (images, point clouds, oxts data) into the output directory with a `scene_info.txt` metadata file.

### 4. Semantic search (optional)

```bash
pip install -e ".[embedding]"
scene-db index --embed              # Build embeddings
scene-db search -s "car turning at intersection"
```

### 5. nuScenes ingestion

```bash
scene-db ingest /path/to/nuscenes --dataset-name nuscenes --nuscenes-version v1.0-mini
```

### 6. Check index status

```bash
scene-db index
```

## CLI Reference

| Command | Description |
|---|---|
| `scene-db ingest <path>` | Ingest a dataset (KITTI or nuScenes) |
| `scene-db index` | Show index status |
| `scene-db index --embed` | Build embedding index for semantic search |
| `scene-db search <query>` | Search scenes by caption text |
| `scene-db search -s <query>` | Semantic search using embeddings |
| `scene-db export --id <id>` | Export scene files to a directory |

### Options

```
scene-db ingest:
  --dataset-name TEXT       Dataset name: kitti or nuscenes [default: kitti]
  --chunk-duration FLOAT    Chunk duration in seconds [default: 5.0]
  --nuscenes-version TEXT   nuScenes version [default: v1.0-mini]
  --db PATH                 Database path [default: ~/.scene-db/scene.db]

scene-db search:
  -s, --semantic            Use semantic search (requires embeddings)
  -k INTEGER                Number of results [default: 10]
  --db PATH                 Database path

scene-db export:
  --id TEXT                 Scene chunk ID (required)
  -o, --output PATH         Output directory [default: ./export]
  --db PATH                 Database path
```

## Supported Datasets

### KITTI

```
<sequence_dir>/
├── oxts/
│   ├── timestamps.txt
│   └── data/
│       ├── 0000000000.txt
│       └── ...
├── image_00/data/
├── image_02/data/
└── velodyne_points/data/
```

Download: https://www.cvlibs.net/datasets/kitti/raw_data.php

### nuScenes

```
<dataroot>/
├── v1.0-mini/         # metadata JSONs
│   ├── scene.json
│   ├── sample.json
│   ├── sample_data.json
│   └── ego_pose.json
├── samples/           # keyframe sensor data
└── sweeps/            # non-keyframe sensor data
```

Download: https://www.nuscenes.org/download

## Architecture

```
Dataset (KITTI / nuScenes)
  → Ingest & Chunk (5s windows)
    → Feature Extraction (speed, distance)
      → Captioning (rule-based or VLM)
        → SQLite Storage
          → Search (text LIKE or semantic embedding)
            → Export
```

- **Storage**: SQLite (`~/.scene-db/scene.db`)
- **Chunking**: Fixed-length time windows (default 5 seconds)
- **Search**: SQL LIKE on caption text, or cosine similarity on embeddings
- **Captioning**: Rule-based (default) or VLM via OpenAI API

## Development

```bash
git clone https://github.com/rsasaki0109/scene-db.git
cd scene-db
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"
pytest
```

## License

MIT
