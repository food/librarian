import subprocess
from pathlib import Path

import pytest

from librarian.bundle.gitops import GitOps
from librarian.bundle.store import BundleStore
from librarian.config import Config, Env, Settings
from librarian.search.fts import FtsIndex


@pytest.fixture
def store(tmp_path: Path) -> BundleStore:
    s = BundleStore(tmp_path / "bundle")
    s.init_layout()
    return s


@pytest.fixture
def git(store: BundleStore) -> GitOps:
    g = GitOps(store.root)
    # local test identity, independent of the global git config
    g.ensure_repo()
    subprocess.run(["git", "-C", str(store.root), "config", "user.name", "test"], check=True)
    subprocess.run(["git", "-C", str(store.root), "config", "user.email", "t@t"], check=True)
    return g


@pytest.fixture
def fts(tmp_path: Path) -> FtsIndex:
    return FtsIndex(tmp_path / "data" / "librarian.db")


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    env = Env(bundle_path=tmp_path / "bundle", data_path=tmp_path / "data",
              client_keys="test:key")
    return Settings(env=env, cfg=Config())
