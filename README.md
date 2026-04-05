# edge-scene-db

**Find the edge cases hiding in your driving logs.**

edge-scene-db ingests autonomous driving and robotics log data, chunks it by time, extracts motion features, and automatically detects localization/perception edge cases.

## Supported Datasets

| Format | Sensors | Command |
|---|---|---|
| **KITTI** | Camera, LiDAR, GPS/IMU | `scene-db ingest /path/to/drive_sync/` |
| **nuScenes** | 6 cameras, LiDAR, RADAR | `scene-db ingest /path/ --dataset-name nuscenes` |
| **rosbag** (GLIM, FAST-LIO, LIO-SAM, etc.) | LiDAR + IMU | `scene-db ingest file.bag` |
| **PPC Dataset** | GNSS + IMU | `scene-db ingest /path/PPC-Dataset/` |

Auto-detection: `.bag` files → rosbag, `oxts/` → KITTI, `v1.0-mini/` → nuScenes, `reference.csv` → PPC.

## Installation

```bash
pip install -e .                    # Core
pip install -e ".[rosbag]"          # + rosbag support (GLIM, FAST-LIO, etc.)
pip install -e ".[embedding]"       # + semantic search
pip install -e ".[vlm]"             # + VLM captioning (OpenAI)
pip install -e ".[all]"             # Everything
```

## Quick Start

```bash
# 1. Ingest data (auto-detects format)
scene-db ingest /path/to/kitti_drive/
scene-db ingest recording.bag
scene-db ingest /path/PPC-Dataset/

# 2. See what you have
scene-db stats

# 3. Detect edge cases automatically
scene-db edge-cases
scene-db edge-cases -c localization --severity critical

# 4. Search with filters
scene-db search "turning"
scene-db search --min-yaw 20 --sort yaw
scene-db search --min-decel 2.0 --sort decel
scene-db search --max-speed 5

# 5. Export a scene
scene-db export --id <scene_id> -o ./output/
```

## Edge Case Detection

`scene-db edge-cases` automatically flags scenes that stress localization or perception systems:

| Category | Rule | Threshold | Severity |
|---|---|---|---|
| **Localization** | High yaw rate | > 20 deg/s | critical |
| **Localization** | High speed (GPS latency) | > 60 km/h | warning/critical |
| **Localization** | Near-zero speed (GPS noise) | < 3 km/h | warning |
| **Localization** | Start from stop | accel + stationary | warning |
| **Both** | Yaw + decel combined | yaw > 10 + decel > 1.0 | critical |
| **Perception** | Hard braking (pitch shift) | > 3.0 m/s² | critical |
| **Perception** | Decel to stop (tracking handoff) | decel > 1 + slow | warning |

```bash
# Filter by category and severity
scene-db edge-cases -c localization --severity critical -n 20
scene-db edge-cases -c perception -n 10
```

## Feature Extraction

Each 5-second scene chunk gets:

| Feature | Description | Edge case relevance |
|---|---|---|
| `avg_speed_kmh` | Average speed | GPS latency, LiDAR distortion |
| `distance_m` | Distance traveled | Dead-reckoning drift |
| `max_accel_ms2` | Peak acceleration | Sensor dynamics |
| `max_decel_ms2` | Peak deceleration | Pitch shift, FOV change |
| `avg_yaw_rate_degs` | Average heading change rate | IMU bias, wheel slip |
| `max_yaw_rate_degs` | Peak heading change rate | EKF heading stress |

Captions auto-generated with keywords: `stationary`, `moving slowly`, `moving forward`, `high speed`, `turning`, `sharp turn`, `gentle curve`, `braking`, `hard braking`.

## CLI Reference

| Command | Description |
|---|---|
| `scene-db ingest <path>` | Ingest dataset (auto-detects format) |
| `scene-db edge-cases` | Detect localization/perception edge cases |
| `scene-db search <query>` | Search by caption text + feature filters |
| `scene-db stats` | Show database statistics |
| `scene-db index [--embed]` | Show index / build embeddings |
| `scene-db export --id <id>` | Export scene files |

### Key Options

```
scene-db ingest:
  --dataset-name TEXT       auto, kitti, nuscenes, rosbag, ppc
  --chunk-duration FLOAT    Chunk duration in seconds [default: 5.0]
  --vlm                     Use VLM captioning (requires OPENAI_API_KEY)
  --imu-topic TEXT          IMU topic for rosbag
  --odom-topic TEXT         Odometry topic for rosbag

scene-db search:
  --min-speed / --max-speed Filter by speed (km/h)
  --min-decel               Filter by deceleration (m/s²)
  --min-yaw                 Filter by yaw rate (deg/s)
  --sort                    Sort by: speed, decel, yaw, accel
  -s, --semantic            Semantic search (requires embeddings)

scene-db edge-cases:
  -c, --category            localization, perception, both
  --severity                critical, warning, info
  -n                        Max results [default: 20]
```

## Tested Datasets

| Source | Type | Sensors | Scenes | Key Edge Cases | Link |
|---|---|---|---|---|---|
| KITTI (25 seq) | Vehicle | LiDAR+Camera+GPS/IMU | 147 | yaw 30°/s, 77 km/h, hard brake | [kitti](https://www.cvlibs.net/datasets/kitti/raw_data.php) |
| nuScenes mini | Vehicle | 6cam+LiDAR+RADAR | 40 | - | [nuscenes](https://www.nuscenes.org/download) |
| GLIM (Ouster) | Handheld | OS1-128 + IMU | 23 | IMU drift (9→588 km/h) | [glim](https://github.com/koide3/glim) |
| Cartographer 3D | Backpack | 2x VLP-16 + IMU | 243 | 20min drift test | [cartographer](https://google-cartographer-ros.readthedocs.io/) |
| PPC Dataset | Vehicle | GNSS + IMU | 2354 | Urban canyon, loop closure | [ppc](https://github.com/taroz/PPC-Dataset) |
| AIST Park | Vehicle | Ouster OS1 + IMU | 29 | **decel 11.2 m/s² (max)** | [zenodo](https://zenodo.org/records/6836915) |
| Flatwall | Handheld | Livox + IMU | 7 | **LiDAR degeneration** | [zenodo](https://zenodo.org/records/7641866) |
| **Total** | | | **2843** | | **99,537 frames** |

## LiDAR SLAM Validation Guide

Which data to use for testing your LiDAR-IMU SLAM / localization system:

| Test Case | Recommended Data | Why |
|---|---|---|
| **Basic sanity check** | GLIM os1_128 (491 MB, 115s) | Small, Ouster, easy to run |
| **Aggressive dynamics** | AIST Park (2.1 GB, 144s) | decel 11.2 m/s², hard braking throughout |
| **LiDAR degeneration** | Flatwall (306 MB, 33s) | Wall-only environment, scan matching fails |
| **Long-term drift** | Cartographer 3D (9.3 GB, 20min) | 20 minutes of continuous operation |
| **Loop closure** | PPC Tokyo run1/run2 | GPS ground truth + loop detection |
| **Urban canyon GNSS** | PPC Nagoya/Tokyo | Multipath, signal blockage |
| **High yaw rate** | KITTI drive_0014 / PPC | yaw 30°/s turning stress |

### Example: Testing with Ouster data (GLIM / AIST Park)

```bash
# Remap topics for your system
ros2 launch your_lio_package lio.launch.xml
# points_raw → /os_cloud_node/points
# imu_raw    → /os_cloud_node/imu
ros2 bag play aist_park_01.bag --clock
```

### Sequence Analysis for SLAM

```bash
scene-db sequences   # distance, duration, loop detection, revisit count
```

```
ppc/tokyo_run1     9.9 km  40 min  ✓ 2m   1386 revisits
ppc/tokyo_run2     6.9 km  30 min  ✓ 1m   1663 revisits
```

## Architecture

```
Raw Data (KITTI / nuScenes / rosbag / PPC)
  → Ingest & Chunk (5-sec time windows)
    → Feature Extraction (speed, distance, yaw rate, acceleration)
      → Captioning (rule-based or VLM)
        → SQLite Storage
          → Edge Case Detection (localization / perception rules)
          → Search (text, filters, or semantic embedding)
            → Export
```

## Development

```bash
git clone https://github.com/rsasaki0109/edge-scene-db.git
cd edge-scene-db
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[test,rosbag]"
pytest  # 120 tests
```

## License

MIT
