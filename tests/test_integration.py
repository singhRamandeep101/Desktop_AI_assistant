"""Integration tests for Friday Assistant."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import AssistantConfig
from src.logger_config import FridayLogger


class TestFridayAssistantIntegration:
    """Integration tests for Friday Assistant."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return AssistantConfig()
    
    @pytest.fixture
    def logger(self):
        """Create test logger."""
        with tempfile.TemporaryDirectory() as temp_dir:
            return FridayLogger(log_dir=temp_dir, log_level="INFO")
    
    @patch('src.transcription_service.WhisperModel')
    def test_full_command_processing(self, mock_whisper_model, config, logger):
        """Test full command processing pipeline."""
        from src.audio_handler import AudioHandler
        from src.command_parser import CommandParser
        from src.transcription_service import TranscriptionService
        from src.tts_service import TTSService
        
        # Mock Whisper model
        mock_model = MagicMock()
        mock_segment = MagicMock()
        mock_segment.text = "what time is it"
        mock_model.transcribe.return_value = ([mock_segment], {})
        mock_whisper_model.return_value = mock_model
        
        # Initialize services
        audio_handler = AudioHandler(config, logger)
        transcription_service = TranscriptionService(config, logger)
        command_parser = CommandParser(logger)
        
        # Create test audio
        test_audio = np.array([0.5, -0.5, 0.3, -0.3] * 1000, dtype=np.float32)
        
        # Preprocess audio
        processed_audio = audio_handler.preprocess_audio(test_audio)
        assert processed_audio.dtype == np.float32
        
        # Validate audio
        assert audio_handler.validate_audio(processed_audio) is True
        
        # Transcribe
        text = transcription_service.transcribe_command(processed_audio)
        assert text == "what time is it"
        
        # Parse command
        command = command_parser.parse_command(text)
        assert command["intent"] == "time_query"
        assert "time" in command["text"].lower()
        
        # Get response
        response = command_parser.get_response(command)
        assert len(response) > 0
        assert "time" in response.lower()
    
    @patch('src.transcription_service.WhisperModel')
    def test_error_recovery(self, mock_whisper_model, config, logger):
        """Test error recovery in transcription."""
        from src.transcription_service import TranscriptionService
        import onnxruntime
        
        # Mock model that raises ONNX error first, then succeeds
        mock_model = MagicMock()
        mock_segment = MagicMock()
        mock_segment.text = "test transcription"
        
        # First call raises error, second succeeds
        mock_model.transcribe.side_effect = [
            onnxruntime.capi.onnxruntime_pybind11_state.InvalidArgument(
                "Unexpected input data type. Actual: (tensor(double)) , expected: (tensor(float))"
            ),
            ([mock_segment], {})
        ]
        mock_whisper_model.return_value = mock_model
        
        transcription_service = TranscriptionService(config, logger)
        test_audio = np.array([0.5, -0.5] * 1000, dtype=np.float32)
        
        # Should retry and succeed
        text, info = transcription_service.transcribe_audio(test_audio)
        assert text == "test transcription"
        assert mock_model.transcribe.call_count == 2
    
    def test_command_intents(self, logger):
        """Test various command intents."""
        from src.command_parser import CommandParser
        
        parser = CommandParser(logger)
        
        test_cases = [
            ("hello", "greeting"),
            ("hi there", "greeting"),
            ("what time is it", "time_query"),
            ("what's the date", "date_query"),
            ("what's the weather", "weather_query"),
            ("exit", "exit"),
            ("quit", "exit"),
            ("help me", "help"),
            ("random text", "general"),
        ]
        
        for text, expected_intent in test_cases:
            command = parser.parse_command(text)
            assert command["intent"] == expected_intent, f"Failed for: {text}"
    
    def test_audio_preprocessing_types(self, config, logger):
        """Test audio preprocessing with different types."""
        from src.audio_handler import AudioHandler
        
        handler = AudioHandler(config, logger)
        
        # Test int16 conversion
        audio_int16 = np.array([1000, -1000, 500], dtype=np.int16)
        processed = handler.preprocess_audio(audio_int16)
        assert processed.dtype == np.float32
        assert np.max(np.abs(processed)) <= 1.0
        
        # Test float64 conversion
        audio_float64 = np.array([0.5, -0.5, 0.25], dtype=np.float64)
        processed = handler.preprocess_audio(audio_float64)
        assert processed.dtype == np.float32
        
        # Test float32 (no conversion needed)
        audio_float32 = np.array([0.5, -0.5, 0.25], dtype=np.float32)
        processed = handler.preprocess_audio(audio_float32)
        assert processed.dtype == np.float32
        assert np.array_equal(processed, audio_float32)
    
    def test_empty_transcription_handling(self, config, logger):
        """Test handling of empty transcription results."""
        from src.transcription_service import TranscriptionService
        from unittest.mock import patch
        
        with patch('src.transcription_service.WhisperModel') as mock_whisper:
            mock_model = MagicMock()
            mock_segment = MagicMock()
            mock_segment.text = ""  # Empty text
            mock_model.transcribe.return_value = ([mock_segment], {})
            mock_whisper.return_value = mock_model
            
            service = TranscriptionService(config, logger)
            test_audio = np.array([0.1] * 1000, dtype=np.float32)
            
            text, info = service.transcribe_audio(test_audio)
            assert text is None  # Should return None for empty transcription


