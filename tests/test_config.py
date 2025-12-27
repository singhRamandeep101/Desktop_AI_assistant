"""Tests for configuration management."""

import json
import os
import tempfile
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    AssistantConfig,
    AudioConfig,
    ErrorHandlingConfig,
    LoggingConfig,
    PerformanceConfig,
    PiperConfig,
    WhisperConfig
)


class TestWhisperConfig:
    """Tests for WhisperConfig."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = WhisperConfig()
        assert config.model == "base"
        assert config.device == "cpu"
        assert config.compute_type == "int8"
        assert config.beam_size == 1
        assert config.vad_filter is True
    
    def test_custom_values(self):
        """Test custom configuration values."""
        config = WhisperConfig(
            model="small",
            device="cuda",
            compute_type="float16",
            beam_size=5,
            vad_filter=False
        )
        assert config.model == "small"
        assert config.device == "cuda"
        assert config.compute_type == "float16"
        assert config.beam_size == 5
        assert config.vad_filter is False


class TestPiperConfig:
    """Tests for PiperConfig."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = PiperConfig()
        assert config.model_path is None
        assert config.config_path is None
        assert config.use_cuda is False
        assert config.voices_dir == "piper_voices"
    
    def test_custom_values(self):
        """Test custom configuration values."""
        config = PiperConfig(
            model_path="test.onnx",
            config_path="test.json",
            use_cuda=True,
            voices_dir="custom_voices"
        )
        assert config.model_path == "test.onnx"
        assert config.config_path == "test.json"
        assert config.use_cuda is True
        assert config.voices_dir == "custom_voices"


class TestAudioConfig:
    """Tests for AudioConfig."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = AudioConfig()
        assert config.sample_rate == 16000
        assert config.chunk_size == 1024
        assert config.channels == 1
        assert config.format == "int16"


class TestAssistantConfig:
    """Tests for AssistantConfig."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = AssistantConfig()
        assert isinstance(config.whisper, WhisperConfig)
        assert isinstance(config.piper, PiperConfig)
        assert isinstance(config.audio, AudioConfig)
        assert isinstance(config.performance, PerformanceConfig)
        assert isinstance(config.error_handling, ErrorHandlingConfig)
        assert isinstance(config.logging, LoggingConfig)
    
    def test_from_dict(self):
        """Test loading config from dictionary."""
        config_dict = {
            "whisper": {"model": "tiny", "beam_size": 2},
            "piper": {"voices_dir": "test_voices"},
            "audio": {"sample_rate": 22050},
            "performance": {"enable_model_caching": False},
            "error_handling": {"max_retries": 5},
            "logging": {"log_level": "DEBUG"}
        }
        
        config = AssistantConfig.from_dict(config_dict)
        assert config.whisper.model == "tiny"
        assert config.whisper.beam_size == 2
        assert config.piper.voices_dir == "test_voices"
        assert config.audio.sample_rate == 22050
        assert config.performance.enable_model_caching is False
        assert config.error_handling.max_retries == 5
        assert config.logging.log_level == "DEBUG"
    
    def test_from_dict_partial(self):
        """Test loading config with partial dictionary."""
        config_dict = {
            "whisper": {"model": "small"}
        }
        
        config = AssistantConfig.from_dict(config_dict)
        assert config.whisper.model == "small"
        # Other values should be defaults
        assert config.whisper.device == "cpu"
        assert config.audio.sample_rate == 16000
    
    def test_from_json_file(self):
        """Test loading config from JSON file."""
        config_dict = {
            "whisper": {"model": "base", "device": "cpu"},
            "piper": {"voices_dir": "piper_voices"},
            "audio": {"sample_rate": 16000},
            "performance": {"enable_model_caching": True},
            "error_handling": {"max_retries": 3},
            "logging": {"log_level": "INFO"}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_dict, f)
            temp_path = f.name
        
        try:
            config = AssistantConfig.from_file(temp_path)
            assert config.whisper.model == "base"
            assert config.whisper.device == "cpu"
        finally:
            os.unlink(temp_path)
    
    def test_from_yaml_file(self):
        """Test loading config from YAML file."""
        try:
            import yaml
        except ImportError:
            pytest.skip("PyYAML not installed")
        
        config_dict = {
            "whisper": {"model": "base"},
            "piper": {"voices_dir": "piper_voices"},
            "audio": {"sample_rate": 16000},
            "performance": {"enable_model_caching": True},
            "error_handling": {"max_retries": 3},
            "logging": {"log_level": "INFO"}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_dict, f)
            temp_path = f.name
        
        try:
            config = AssistantConfig.from_file(temp_path)
            assert config.whisper.model == "base"
        finally:
            os.unlink(temp_path)
    
    def test_from_file_not_found(self):
        """Test loading config from non-existent file."""
        with pytest.raises(FileNotFoundError):
            AssistantConfig.from_file("nonexistent_config.yaml")
    
    def test_from_env(self):
        """Test loading config from environment variables."""
        # Set environment variables
        os.environ["WHISPER_MODEL"] = "small"
        os.environ["WHISPER_DEVICE"] = "cuda"
        os.environ["LOG_LEVEL"] = "DEBUG"
        
        try:
            config = AssistantConfig.from_env()
            assert config.whisper.model == "small"
            assert config.whisper.device == "cuda"
            assert config.logging.log_level == "DEBUG"
        finally:
            # Clean up
            os.environ.pop("WHISPER_MODEL", None)
            os.environ.pop("WHISPER_DEVICE", None)
            os.environ.pop("LOG_LEVEL", None)
    
    def test_load_with_file(self):
        """Test load method with file path."""
        config_dict = {
            "whisper": {"model": "base"},
            "piper": {"voices_dir": "piper_voices"},
            "audio": {"sample_rate": 16000},
            "performance": {"enable_model_caching": True},
            "error_handling": {"max_retries": 3},
            "logging": {"log_level": "INFO"}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_dict, f)
            temp_path = f.name
        
        try:
            config = AssistantConfig.load(temp_path)
            assert config.whisper.model == "base"
        finally:
            os.unlink(temp_path)
    
    def test_load_without_file(self):
        """Test load method without file (uses defaults/env)."""
        config = AssistantConfig.load()
        assert isinstance(config, AssistantConfig)
        assert config.whisper.model == "base"
    
    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = AssistantConfig()
        config_dict = config.to_dict()
        
        assert "whisper" in config_dict
        assert "piper" in config_dict
        assert "audio" in config_dict
        assert "performance" in config_dict
        assert "error_handling" in config_dict
        assert "logging" in config_dict
    
    def test_save_yaml(self):
        """Test saving config to YAML file."""
        try:
            import yaml
        except ImportError:
            pytest.skip("PyYAML not installed")
        
        config = AssistantConfig()
        config.whisper.model = "test_model"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name
        
        try:
            config.save(temp_path)
            assert Path(temp_path).exists()
            
            # Verify it can be loaded back
            loaded = AssistantConfig.from_file(temp_path)
            assert loaded.whisper.model == "test_model"
        finally:
            if Path(temp_path).exists():
                os.unlink(temp_path)
    
    def test_config_merge_non_none(self):
        """Test that None values in file don't overwrite env values."""
        # Set env var
        os.environ["WHISPER_MODEL"] = "env_model"
        
        config_dict = {
            "whisper": {"model": None},  # None should not overwrite env
            "piper": {"voices_dir": "file_voices"}  # This should be used
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_dict, f)
            temp_path = f.name
        
        try:
            config = AssistantConfig.load(temp_path)
            # Model should come from env (not None from file)
            assert config.whisper.model == "env_model"
            # Voices dir should come from file
            assert config.piper.voices_dir == "file_voices"
        finally:
            os.environ.pop("WHISPER_MODEL", None)
            os.unlink(temp_path)


