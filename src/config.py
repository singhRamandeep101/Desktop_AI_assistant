import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union


@dataclass
class WhisperConfig:
    model: str = "base"
    device: str = "cpu"
    compute_type: str = "int8"
    beam_size: int = 1
    vad_filter: bool = True


@dataclass
class PiperConfig:
    model_path: Optional[str] = None
    config_path: Optional[str] = None
    use_cuda: bool = False
    voices_dir: str = "piper_voices"


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    chunk_size: int = 1024
    channels: int = 1
    format: str = "int16"


@dataclass
class PerformanceConfig:
    enable_model_caching: bool = True
    transcription_timeout: float = 30.0


@dataclass
class ErrorHandlingConfig:
    max_retries: int = 3
    retry_delay: float = 0.5


@dataclass
class LoggingConfig:
    log_dir: str = "logs"
    log_level: str = "INFO"


@dataclass
class AssistantConfig:
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    piper: PiperConfig = field(default_factory=PiperConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    error_handling: ErrorHandlingConfig = field(default_factory=ErrorHandlingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "AssistantConfig":
        """Create config from dictionary."""
        return cls(
            whisper=WhisperConfig(**config_dict.get("whisper", {})),
            piper=PiperConfig(**config_dict.get("piper", {})),
            audio=AudioConfig(**config_dict.get("audio", {})),
            performance=PerformanceConfig(**config_dict.get("performance", {})),
            error_handling=ErrorHandlingConfig(**config_dict.get("error_handling", {})),
            logging=LoggingConfig(**config_dict.get("logging", {}))
        )
    
    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> "AssistantConfig":
        """Load config from YAML or JSON file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, "r", encoding="utf-8") as f:
            if config_path.suffix.lower() == ".yaml" or config_path.suffix.lower() == ".yml":
                try:
                    import yaml
                    config_dict = yaml.safe_load(f)
                except ImportError:
                    raise ImportError("PyYAML is required for YAML config files. Install with: pip install pyyaml")
            else:
                config_dict = json.load(f)
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_env(cls) -> "AssistantConfig":
        """Load config from environment variables."""
        config_dict = {}
        
        # Whisper settings
        if os.getenv("WHISPER_MODEL"):
            config_dict.setdefault("whisper", {})["model"] = os.getenv("WHISPER_MODEL")
        if os.getenv("WHISPER_DEVICE"):
            config_dict.setdefault("whisper", {})["device"] = os.getenv("WHISPER_DEVICE")
        if os.getenv("WHISPER_COMPUTE_TYPE"):
            config_dict.setdefault("whisper", {})["compute_type"] = os.getenv("WHISPER_COMPUTE_TYPE")
        
        # Piper settings
        if os.getenv("PIPER_MODEL_PATH"):
            config_dict.setdefault("piper", {})["model_path"] = os.getenv("PIPER_MODEL_PATH")
        if os.getenv("PIPER_VOICES_DIR"):
            config_dict.setdefault("piper", {})["voices_dir"] = os.getenv("PIPER_VOICES_DIR")
        
        # Logging
        if os.getenv("LOG_LEVEL"):
            config_dict.setdefault("logging", {})["log_level"] = os.getenv("LOG_LEVEL")
        if os.getenv("LOG_DIR"):
            config_dict.setdefault("logging", {})["log_dir"] = os.getenv("LOG_DIR")
        
        if config_dict:
            return cls.from_dict(config_dict)
        return cls()
    
    @classmethod
    def load(cls, config_path: Optional[Union[str, Path]] = None) -> "AssistantConfig":
        """Load config from file or environment, with fallback to defaults."""
        # Try environment variables first
        config = cls.from_env()
        
        # Override with file if provided
        if config_path:
            try:
                file_config = cls.from_file(config_path)
                # Merge file config over env config (file takes precedence, but only non-None values)
                def merge_dicts(env_dict: dict, file_dict: dict) -> dict:
                    """Merge file dict over env dict, but skip None values from file."""
                    merged = env_dict.copy()
                    for k, v in file_dict.items():
                        if v is not None:  # Only override if file value is not None
                            merged[k] = v
                    return merged
                
                config = cls.from_dict({
                    "whisper": merge_dicts(config.whisper.__dict__, file_config.whisper.__dict__),
                    "piper": merge_dicts(config.piper.__dict__, file_config.piper.__dict__),
                    "audio": merge_dicts(config.audio.__dict__, file_config.audio.__dict__),
                    "performance": merge_dicts(config.performance.__dict__, file_config.performance.__dict__),
                    "error_handling": merge_dicts(config.error_handling.__dict__, file_config.error_handling.__dict__),
                    "logging": merge_dicts(config.logging.__dict__, file_config.logging.__dict__)
                })
            except FileNotFoundError:
                # If file doesn't exist, just use env/defaults
                pass
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "whisper": self.whisper.__dict__,
            "piper": self.piper.__dict__,
            "audio": self.audio.__dict__,
            "performance": self.performance.__dict__,
            "error_handling": self.error_handling.__dict__,
            "logging": self.logging.__dict__
        }
    
    def save(self, config_path: Union[str, Path]):
        """Save config to YAML file."""
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML is required to save YAML config files. Install with: pip install pyyaml")
        
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

