import importlib
import sys

import pytest
from starlette.testclient import TestClient

from librarian.config import get_settings


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Server app behind BearerAuth, on a throwaway bundle."""
    monkeypatch.setenv("BUNDLE_PATH", str(tmp_path / "bundle"))
    monkeypatch.setenv("DATA_PATH", str(tmp_path / "data"))
    monkeypatch.setenv("DOCS_PATH", str(tmp_path / "docs"))
    get_settings.cache_clear()
    # the module builds its ops at import time, so it has to load under this env
    monkeypatch.delitem(sys.modules, "librarian.server", raising=False)
    server = importlib.import_module("librarian.server")
    app = server.BearerAuth(server.mcp.http_app(path="/mcp"), {"key": "test"})
    yield TestClient(app)
    get_settings.cache_clear()


def test_browser_requests_need_no_key(client):
    r = client.get("/", follow_redirects=False)
    assert r.status_code == 307
    assert r.headers["location"] == "/ui"
    assert client.get("/favicon.ico").status_code == 404


def test_api_still_needs_key(client):
    assert client.get("/api/graph").status_code == 401
    r = client.get("/api/graph", headers={"Authorization": "Bearer key"})
    assert r.status_code == 200
