"""Tests for audio handler."""

import numpy as np
import pytest


@pytest.fixture
def mock_config():
    """Create mock config for testing."""
    from dataclasses import dataclass
    
    @dataclass
    class MockAudioConfig:
        sample_rate: int = 16000
        chunk_size: int = 1024
        channels: int = 1
        format: str = "int16"
    
    @dataclass
    class MockConfig:
        audio: MockAudioConfig = MockAudioConfig()
    
    return MockConfig()


@pytest.fixture
def mock_logger():
    """Create mock logger for testing."""
    class MockLogger:
        def log_info(self, msg):
            pass
        
        def log_error(self, msg, exc_info=False):
            pass
        
        def log_warning(self, msg):
            pass
        
        def log_debug(self, msg):
            pass
    
    return MockLogger()


def test_preprocess_audio(mock_config, mock_logger):
    """Test audio preprocessing."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    
    from audio_handler import AudioHandler
    
    handler = AudioHandler(mock_config, mock_logger)
    
    # Test int16 to float32 conversion
    audio_int16 = np.array([1000, -1000, 500], dtype=np.int16)
    audio_processed = handler.preprocess_audio(audio_int16)
    
    assert audio_processed.dtype == np.float32
    assert np.max(np.abs(audio_processed)) <= 1.0
    
    # Test float64 to float32 conversion
    audio_float64 = np.array([0.5, -0.5, 0.25], dtype=np.float64)
    audio_processed = handler.preprocess_audio(audio_float64)
    
    assert audio_processed.dtype == np.float32


def test_validate_audio(mock_config, mock_logger):
    """Test audio validation."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    
    from audio_handler import AudioHandler
    
    handler = AudioHandler(mock_config, mock_logger)
    
    # Valid audio
    valid_audio = np.array([0.5, -0.5, 0.25], dtype=np.float32)
    assert handler.validate_audio(valid_audio) == True
    
    # Empty audio
    empty_audio = np.array([], dtype=np.float32)
    assert handler.validate_audio(empty_audio) == False
    
    # Silent audio
    silent_audio = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    assert handler.validate_audio(silent_audio) == False
    
    # None audio
    assert handler.validate_audio(None) == False


