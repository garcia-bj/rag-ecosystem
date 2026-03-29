"""pytest configuration — path setup, env loading, and shared fixtures."""

import sys
from pathlib import Path

# Make ecosystem/ the root so `import ingestion` works without install
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

# Load .env so integration tests have QDRANT_URL, NEO4J_PASSWORD, etc.
from dotenv import load_dotenv
load_dotenv(_ROOT / ".env")

import pytest


@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures"
