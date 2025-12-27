"""Tests for transcription service."""

import numpy as np
import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import AssistantConfig
from src.logger_config import FridayLogger


@pytest.fixture
def mock_config():
    """Create mock config for testing."""
    return AssistantConfig()


@pytest.fixture
def mock_logger():
    """Create mock logger for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        return FridayLogger(log_dir=temp_dir, log_level="INFO")


@patch('src.transcription_service.WhisperModel')
def test_load_model(mock_whisper_model, mock_config, mock_logger):
    """Test model loading."""
    from src.transcription_service import TranscriptionService
    
    mock_model = MagicMock()
    mock_whisper_model.return_value = mock_model
    
    service = TranscriptionService(mock_config, mock_logger)
    result = service.load_model()
    
    assert result is True
    assert service.model is not None
    mock_whisper_model.assert_called_once()


@patch('src.transcription_service.WhisperModel')
def test_transcribe_audio_success(mock_whisper_model, mock_config, mock_logger):
    """Test successful transcription."""
    from src.transcription_service import TranscriptionService
    
    mock_model = MagicMock()
    mock_segment = MagicMock()
    mock_segment.text = "test transcription"
    mock_model.transcribe.return_value = ([mock_segment], {})
    mock_whisper_model.return_value = mock_model
    
    service = TranscriptionService(mock_config, mock_logger)
    test_audio = np.array([0.5, -0.5] * 1000, dtype=np.float32)
    
    text, info = service.transcribe_audio(test_audio)
    
    assert text == "test transcription"
    assert service.model is not None


def test_ensure_float32(mock_config, mock_logger):
    """Test float32 conversion."""
    from src.transcription_service import TranscriptionService
    
    service = TranscriptionService(mock_config, mock_logger)
    
    # Test with int16 audio
    audio_int16 = np.array([1000, -1000, 500], dtype=np.int16)
    audio_float32 = service._ensure_float32(audio_int16)
    
    assert audio_float32.dtype == np.float32
    
    # Test with float64 audio
    audio_float64 = np.array([0.5, -0.5, 0.25], dtype=np.float64)
    audio_float32 = service._ensure_float32(audio_float64)
    
    assert audio_float32.dtype == np.float32
    
    # Test with float32 (should not change)
    audio_float32_input = np.array([0.5, -0.5, 0.25], dtype=np.float32)
    audio_float32_output = service._ensure_float32(audio_float32_input)
    
    assert audio_float32_output.dtype == np.float32
    assert np.array_equal(audio_float32_input, audio_float32_output)


def test_validate_audio(mock_config, mock_logger):
    """Test audio validation."""
    from src.transcription_service import TranscriptionService
    
    service = TranscriptionService(mock_config, mock_logger)
    
    # Valid audio
    valid_audio = np.array([0.5, -0.5, 0.25], dtype=np.float32)
    assert service._validate_audio(valid_audio) is True
    
    # Empty audio
    empty_audio = np.array([], dtype=np.float32)
    assert service._validate_audio(empty_audio) is False
    
    # Silent audio
    silent_audio = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    assert service._validate_audio(silent_audio) is False


@patch('src.transcription_service.WhisperModel')
def test_transcribe_command(mock_whisper_model, mock_config, mock_logger):
    """Test transcribe_command method."""
    from src.transcription_service import TranscriptionService
    
    mock_model = MagicMock()
    mock_segment = MagicMock()
    mock_segment.text = "test command"
    mock_model.transcribe.return_value = ([mock_segment], {})
    mock_whisper_model.return_value = mock_model
    
    service = TranscriptionService(mock_config, mock_logger)
    test_audio = np.array([0.5, -0.5] * 1000, dtype=np.float32)
    
    text = service.transcribe_command(test_audio)
    
    assert text == "test command"


@patch('src.transcription_service.WhisperModel')
def test_transcribe_empty_result(mock_whisper_model, mock_config, mock_logger):
    """Test transcription with empty result."""
    from src.transcription_service import TranscriptionService
    
    mock_model = MagicMock()
    mock_segment = MagicMock()
    mock_segment.text = ""  # Empty text
    mock_model.transcribe.return_value = ([mock_segment], {})
    mock_whisper_model.return_value = mock_model
    
    service = TranscriptionService(mock_config, mock_logger)
    test_audio = np.array([0.1] * 1000, dtype=np.float32)
    
    text, info = service.transcribe_audio(test_audio)
    
    assert text is None

