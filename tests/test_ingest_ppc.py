"""Tests for scene_db.ingest_ppc."""

import math
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from scene_db.ingest_ppc import (
    _gps_tow_to_datetime,
    parse_reference_csv,
    ingest_ppc_run,
    ingest_ppc,
)


class TestGpsTowToDatetime:
    def test_epoch(self):
        # GPS epoch: Jan 6, 1980
        result = _gps_tow_to_datetime(0.0, 0)
        assert result == datetime(1980, 1, 6)

    def test_one_week(self):
        result = _gps_tow_to_datetime(0.0, 1)
        assert result == datetime(1980, 1, 13)

    def test_with_seconds(self):
        result = _gps_tow_to_datetime(3600.0, 0)
        assert result == datetime(1980, 1, 6, 1, 0, 0)

    def test_known_gps_week(self):
        # GPS week 2000 starts around 2018-05-27
        result = _gps_tow_to_datetime(0.0, 2000)
        expected = datetime(1980, 1, 6) + timedelta(weeks=2000)
        assert result == expected


class TestParseReferenceCsv:
    def test_parse_basic_csv(self, tmp_path):
        csv_file = tmp_path / "reference.csv"
        # Header + 2 rows with 14 columns
        csv_file.write_text(
            "tow,week,lat,lon,alt,e,n,u,roll,pitch,heading,ve,vn,vu\n"
            "100.0,2000,35.0,139.0,50.0,0,0,0,0.0,0.0,90.0,1.0,2.0,0.5\n"
            "101.0,2000,35.001,139.001,50.5,0,0,0,1.0,0.5,91.0,1.5,2.5,0.3\n"
        )
        records = parse_reference_csv(csv_file)
        assert len(records) == 2
        assert records[0].lat == 35.0
        assert records[0].lon == 139.0
        # heading 90 -> yaw = radians(90 - 90) = 0
        assert records[0].yaw == pytest.approx(0.0)
        # vf = sqrt(ve^2 + vn^2) = sqrt(1 + 4) = sqrt(5)
        assert records[0].vf == pytest.approx(math.sqrt(5.0))

    def test_parse_empty_csv(self, tmp_path):
        csv_file = tmp_path / "reference.csv"
        csv_file.write_text("tow,week,lat,lon,alt,e,n,u,roll,pitch,heading,ve,vn,vu\n")
        records = parse_reference_csv(csv_file)
        assert records == []

    def test_skips_short_rows(self, tmp_path):
        csv_file = tmp_path / "reference.csv"
        csv_file.write_text(
            "tow,week,lat,lon,alt,e,n,u,roll,pitch,heading,ve,vn,vu\n"
            "100.0,2000,35.0\n"  # too short, should be skipped
            "101.0,2000,35.0,139.0,50.0,0,0,0,0.0,0.0,90.0,1.0,2.0,0.5\n"
        )
        records = parse_reference_csv(csv_file)
        assert len(records) == 1


class TestIngestPpcRun:
    def _write_reference_csv(self, run_dir: Path, n_records: int = 20):
        """Write a minimal reference.csv with enough records for chunking."""
        ref = run_dir / "reference.csv"
        lines = ["tow,week,lat,lon,alt,e,n,u,roll,pitch,heading,ve,vn,vu"]
        for i in range(n_records):
            tow = 100.0 + i * 0.5  # 2 Hz
            lat = 35.0 + i * 0.0001
            lon = 139.0 + i * 0.0001
            lines.append(
                f"{tow},2000,{lat},{lon},50.0,0,0,0,0.0,0.0,90.0,1.0,2.0,0.5"
            )
        ref.write_text("\n".join(lines) + "\n")

    def test_ingest_creates_chunks(self, tmp_path):
        run_dir = tmp_path / "tokyo" / "run1"
        run_dir.mkdir(parents=True)
        self._write_reference_csv(run_dir, n_records=30)
        db_path = tmp_path / "test.db"
        n = ingest_ppc_run(run_dir, db_path=db_path)
        assert n > 0

    def test_missing_reference_csv(self, tmp_path):
        run_dir = tmp_path / "empty_run"
        run_dir.mkdir()
        with pytest.raises(FileNotFoundError):
            ingest_ppc_run(run_dir, db_path=tmp_path / "test.db")

    def test_empty_reference_csv(self, tmp_path):
        run_dir = tmp_path / "tokyo" / "run1"
        run_dir.mkdir(parents=True)
        ref = run_dir / "reference.csv"
        ref.write_text("tow,week,lat,lon,alt,e,n,u,roll,pitch,heading,ve,vn,vu\n")
        db_path = tmp_path / "test.db"
        n = ingest_ppc_run(run_dir, db_path=db_path)
        assert n == 0


class TestIngestPpc:
    def _create_dataset(self, base: Path, cities_runs: dict[str, list[str]]):
        """Create directory structure with reference.csv files."""
        for city, runs in cities_runs.items():
            for run in runs:
                run_dir = base / city / run
                run_dir.mkdir(parents=True)
                lines = ["tow,week,lat,lon,alt,e,n,u,roll,pitch,heading,ve,vn,vu"]
                for i in range(20):
                    tow = 100.0 + i * 0.5
                    lat = 35.0 + i * 0.0001
                    lon = 139.0 + i * 0.0001
                    lines.append(
                        f"{tow},2000,{lat},{lon},50.0,0,0,0,0.0,0.0,90.0,1.0,2.0,0.5"
                    )
                (run_dir / "reference.csv").write_text("\n".join(lines) + "\n")

    def test_multi_city(self, tmp_path):
        ds_dir = tmp_path / "ppc_dataset"
        ds_dir.mkdir()
        self._create_dataset(ds_dir, {"tokyo": ["run1", "run2"], "nagoya": ["run1"]})
        db_path = tmp_path / "test.db"
        total = ingest_ppc(ds_dir, db_path=db_path)
        assert total > 0

    def test_empty_dataset(self, tmp_path):
        ds_dir = tmp_path / "empty"
        ds_dir.mkdir()
        db_path = tmp_path / "test.db"
        total = ingest_ppc(ds_dir, db_path=db_path)
        assert total == 0
