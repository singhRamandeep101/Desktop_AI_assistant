"""Tests for TTS service."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import AssistantConfig, PiperConfig
from src.logger_config import FridayLogger


class TestTTSService:
    """Tests for TTSService."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock config."""
        config = AssistantConfig()
        config.piper = PiperConfig(
            voices_dir="piper_voices",
            model_path=None,
            config_path=None
        )
        return config
    
    @pytest.fixture
    def mock_logger(self):
        """Create mock logger."""
        with tempfile.TemporaryDirectory() as temp_dir:
            return FridayLogger(log_dir=temp_dir, log_level="INFO")
    
    def test_initialization_without_piper(self, mock_config, mock_logger):
        """Test TTS service initialization when Piper is not available."""
        with patch('src.tts_service.PIPER_AVAILABLE', False):
            from src.tts_service import TTSService
            
            service = TTSService(mock_config, mock_logger)
            assert service.voice is None
            assert service.is_available() is False
    
    def test_load_voice_no_model(self, mock_config, mock_logger):
        """Test loading voice when no model is available."""
        with patch('src.tts_service.PIPER_AVAILABLE', True):
            with patch('src.tts_service.PiperVoice') as mock_piper:
                from src.tts_service import TTSService
                
                # Mock voices directory with no models
                mock_config.piper.voices_dir = str(Path(__file__).parent / "nonexistent")
                
                service = TTSService(mock_config, mock_logger)
                result = service.load_voice()
                
                assert result is False
    
    @patch('src.tts_service.PIPER_AVAILABLE', True)
    @patch('src.tts_service.PiperVoice')
    def test_load_voice_success(self, mock_piper_voice, mock_config, mock_logger):
        """Test successful voice loading."""
        from src.tts_service import TTSService
        
        # Create temporary voice files
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "test.onnx"
            config_path = Path(temp_dir) / "test.onnx.json"
            
            model_path.touch()
            config_path.write_text('{"test": "config"}')
            
            mock_config.piper.model_path = str(model_path)
            mock_config.piper.config_path = str(config_path)
            
            # Mock PiperVoice.load
            mock_voice = MagicMock()
            mock_piper_voice.load.return_value = mock_voice
            
            service = TTSService(mock_config, mock_logger)
            result = service.load_voice()
            
            assert result is True
            assert service.voice == mock_voice
            mock_piper_voice.load.assert_called_once()
    
    def test_speak_without_voice(self, mock_config, mock_logger):
        """Test speaking when voice is not loaded."""
        with patch('src.tts_service.PIPER_AVAILABLE', False):
            from src.tts_service import TTSService
            
            service = TTSService(mock_config, mock_logger)
            result = service.speak("Test text")
            
            assert result is False
    
    def test_speak_empty_text(self, mock_config, mock_logger):
        """Test speaking with empty text."""
        with patch('src.tts_service.PIPER_AVAILABLE', True):
            with patch('src.tts_service.PiperVoice') as mock_piper:
                from src.tts_service import TTSService
                
                mock_voice = MagicMock()
                service = TTSService(mock_config, mock_logger)
                service.voice = mock_voice
                
                result = service.speak("")
                assert result is False
                
                result = service.speak("   ")
                assert result is False
    
    @patch('src.tts_service.PIPER_AVAILABLE', True)
    @patch('src.tts_service.PiperVoice')
    def test_speak_success(self, mock_piper_voice, mock_config, mock_logger):
        """Test successful speech synthesis."""
        from src.tts_service import TTSService
        import wave
        
        mock_voice = MagicMock()
        service = TTSService(mock_config, mock_logger)
        service.voice = mock_voice
        
        # Test with output path
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            output_path = f.name
        
        try:
            result = service.speak("Test text", output_path=output_path)
            assert result is True
            assert Path(output_path).exists()
            mock_voice.synthesize.assert_called_once()
        finally:
            if Path(output_path).exists():
                Path(output_path).unlink()
    
    def test_is_available(self, mock_config, mock_logger):
        """Test availability check."""
        with patch('src.tts_service.PIPER_AVAILABLE', True):
            with patch('src.tts_service.PiperVoice') as mock_piper:
                from src.tts_service import TTSService
                
                service = TTSService(mock_config, mock_logger)
                assert service.is_available() is True
        
        with patch('src.tts_service.PIPER_AVAILABLE', False):
            from src.tts_service import TTSService
            
            service = TTSService(mock_config, mock_logger)
            assert service.is_available() is False


