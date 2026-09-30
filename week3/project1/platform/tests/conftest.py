"""Shared fixtures: a fresh platform (own data folder, demo mode) and an API client."""

import sys
from pathlib import Path

import pytest

PLATFORM = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PLATFORM))


@pytest.fixture(scope="session")
def sample_dir(tmp_path_factory):
    """12 placeholder single-tile photos with ground truth (generated once per test run)."""
    from backend.sample_tiles import write_folder
    out = tmp_path_factory.mktemp("tiles")
    write_folder(str(out), n=12, seed=5)
    return out


@pytest.fixture
def platform(tmp_path, monkeypatch, sample_dir):
    monkeypatch.delenv("TILE_PLATFORM_CHECKPOINT", raising=False)
    monkeypatch.setenv("TILE_PLATFORM_IMAGES", str(sample_dir / "original"))
    from backend.core import Platform
    return Platform(data_dir=tmp_path / "data", checkpoint="")      # "" = demo mode


@pytest.fixture
def client(platform, tmp_path):
    from fastapi.testclient import TestClient
    from backend.app import create_app
    app = create_app(platform, frontend_dist=tmp_path / "no-frontend")
    with TestClient(app) as c:
        yield c


@pytest.fixture
def photo(sample_dir):
    """Path + ground truth of a damaged sample (usable fraction well below 1)."""
    import pandas as pd
    gt = pd.read_csv(sample_dir / "ground_truth.csv")
    row = gt[(gt.usable_fraction > 0.4) & (gt.usable_fraction < 0.85)].iloc[0]
    return sample_dir / "original" / row.filename, float(row.usable_fraction)
