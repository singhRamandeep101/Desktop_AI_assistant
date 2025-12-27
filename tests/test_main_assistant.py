"""Tests for main FridayAssistant class."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import AssistantConfig
from friday_assistant import FridayAssistant


class TestFridayAssistant:
    """Tests for FridayAssistant main class."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return AssistantConfig()
    
    @patch('src.transcription_service.WhisperModel')
    @patch('src.tts_service.PIPER_AVAILABLE', False)
    def test_initialization(self, mock_whisper_model, config):
        """Test assistant initialization."""
        mock_model = MagicMock()
        mock_whisper_model.return_value = mock_model
        
        assistant = FridayAssistant(config)
        assert assistant.config == config
        assert assistant.logger is not None
        assert assistant.audio_handler is not None
        assert assistant.transcription_service is not None
        assert assistant.tts_service is not None
        assert assistant.command_parser is not None
    
    @patch('src.transcription_service.WhisperModel')
    @patch('src.tts_service.PIPER_AVAILABLE', False)
    def test_initialize_success(self, mock_whisper_model, config):
        """Test successful initialization."""
        mock_model = MagicMock()
        mock_whisper_model.return_value = mock_model
        
        assistant = FridayAssistant(config)
        result = assistant.initialize()
        
        assert result is True
    
    @patch('src.transcription_service.WhisperModel')
    @patch('src.tts_service.PIPER_AVAILABLE', False)
    def test_initialize_failure(self, mock_whisper_model, config):
        """Test initialization failure."""
        mock_whisper_model.side_effect = Exception("Model load failed")
        
        assistant = FridayAssistant(config)
        result = assistant.initialize()
        
        assert result is False
    
    @patch('src.transcription_service.WhisperModel')
    @patch('src.tts_service.PIPER_AVAILABLE', False)
    def test_process_command_with_audio(self, mock_whisper_model, config):
        """Test processing command with provided audio."""
        mock_model = MagicMock()
        mock_segment = MagicMock()
        mock_segment.text = "hello"
        mock_model.transcribe.return_value = ([mock_segment], {})
        mock_whisper_model.return_value = mock_model
        
        assistant = FridayAssistant(config)
        assistant.initialize()
        
        test_audio = np.array([0.5, -0.5] * 1000, dtype=np.float32)
        command = assistant.process_command(audio=test_audio)
        
        assert command is not None
        assert command["intent"] == "greeting"
        assert "hello" in command["text"].lower()
    
    @patch('src.transcription_service.WhisperModel')
    @patch('src.tts_service.PIPER_AVAILABLE', False)
    @patch('src.audio_handler.AUDIO_AVAILABLE', False)
    def test_process_command_no_audio(self, mock_whisper_model, config):
        """Test processing command when audio recording is not available."""
        mock_model = MagicMock()
        mock_whisper_model.return_value = mock_model
        
        assistant = FridayAssistant(config)
        assistant.initialize()
        
        # Should return None when audio recording fails
        command = assistant.process_command()
        assert command is None
    
    @patch('src.transcription_service.WhisperModel')
    @patch('src.tts_service.PIPER_AVAILABLE', False)
    def test_handle_command(self, mock_whisper_model, config):
        """Test command handling."""
        mock_model = MagicMock()
        mock_whisper_model.return_value = mock_model
        
        assistant = FridayAssistant(config)
        assistant.initialize()
        
        command = {
            "intent": "greeting",
            "text": "hello"
        }
        
        # Should not raise exception
        assistant._handle_command(command)
    
    @patch('src.transcription_service.WhisperModel')
    @patch('src.tts_service.PIPER_AVAILABLE', False)
    def test_cleanup(self, mock_whisper_model, config):
        """Test cleanup."""
        mock_model = MagicMock()
        mock_whisper_model.return_value = mock_model
        
        assistant = FridayAssistant(config)
        assistant.initialize()
        
        # Should not raise exception
        assistant.cleanup()
    
    @patch('src.transcription_service.WhisperModel')
    @patch('src.tts_service.PIPER_AVAILABLE', False)
    def test_exit_command(self, mock_whisper_model, config):
        """Test exit command handling."""
        mock_model = MagicMock()
        mock_segment = MagicMock()
        mock_segment.text = "exit"
        mock_model.transcribe.return_value = ([mock_segment], {})
        mock_whisper_model.return_value = mock_model
        
        assistant = FridayAssistant(config)
        assistant.initialize()
        
        test_audio = np.array([0.5, -0.5] * 1000, dtype=np.float32)
        command = assistant.process_command(audio=test_audio)
        
        assert command is not None
        assert command["intent"] == "exit"


