"""
Shared test fixtures and configuration for the KAIROS test suite.

Sets up:
  - sys.path so `from memory.sqlite_store import ...` just works
  - DEBUG logging with timestamps so you see what happened on failure
  - Reusable fixtures: tmp data dirs, in-memory DBs, mock LLM client
"""

import os
import sys
import tempfile

import pytest

# ── path setup ────────────────────────────────────────────────────────────────
# Add runtime/ and project root to sys.path so all imports match production code.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUNTIME_DIR = os.path.join(PROJECT_ROOT, "runtime")
sys.path.insert(0, RUNTIME_DIR)
sys.path.insert(0, PROJECT_ROOT)


# ── logging ───────────────────────────────────────────────────────────────────
# Every test gets timestamped DEBUG output — invaluable when debugging failures.
from config.logging_config import setup_logging
setup_logging(level_override="DEBUG")


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_data_dir(tmp_path):
    """
    Provide a temporary data directory and patch DATA_DIR env var.
    Every test that touches the filesystem uses this — never production data/.
    """
    data_dir = str(tmp_path / "data")
    os.makedirs(data_dir, exist_ok=True)
    old = os.environ.get("DATA_DIR")
    os.environ["DATA_DIR"] = data_dir
    yield data_dir
    if old is None:
        os.environ.pop("DATA_DIR", None)
    else:
        os.environ["DATA_DIR"] = old


@pytest.fixture
def mock_llm_response():
    """
    Factory fixture — returns a function that creates a mock LLMClient
    whose complete() returns the given text.
    """
    from unittest.mock import AsyncMock, MagicMock

    def _make(response_text: str):
        client = MagicMock()
        client.complete = AsyncMock(return_value=response_text)
        client.stream = AsyncMock()
        client.aclose = AsyncMock()
        return client

    return _make
