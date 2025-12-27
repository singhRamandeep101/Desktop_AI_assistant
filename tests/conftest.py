"""Pytest configuration and shared fixtures."""

import tempfile
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import AssistantConfig
from src.logger_config import FridayLogger


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def default_config():
    """Create a default AssistantConfig for testing."""
    return AssistantConfig()


@pytest.fixture
def test_logger(temp_dir):
    """Create a FridayLogger for testing."""
    return FridayLogger(log_dir=str(temp_dir), log_level="INFO")


@pytest.fixture
def test_audio_float32():
    """Create test audio array (float32)."""
    import numpy as np
    return np.array([0.5, -0.5, 0.3, -0.3] * 1000, dtype=np.float32)


@pytest.fixture
def test_audio_int16():
    """Create test audio array (int16)."""
    import numpy as np
    return np.array([1000, -1000, 500, -500] * 1000, dtype=np.int16)


