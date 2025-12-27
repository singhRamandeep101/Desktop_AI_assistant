import sys
import wave
from pathlib import Path
from typing import Any, Optional

PIPER_AVAILABLE = False
PiperVoice = None

try:
    from piper.voice import PiperVoice
    PIPER_AVAILABLE = True
except ImportError:
    piper_path = Path(__file__).parent.parent / "piper-master" / "src" / "python_run"
    if piper_path.exists():
        sys.path.insert(0, str(piper_path))
        try:
            from piper.voice import PiperVoice
            PIPER_AVAILABLE = True
        except ImportError:
            pass


class TTSService:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.voice: Optional[Any] = None
        
        if not PIPER_AVAILABLE or PiperVoice is None:
            self.logger.log_warning("Piper TTS not available - TTS will be disabled")
    
    def load_voice(self) -> bool:
        if not PIPER_AVAILABLE or PiperVoice is None:
            self.logger.log_warning("Piper TTS not available")
            return False
        
        try:
            voices_dir = Path(self.config.piper.voices_dir)
            
            if self.config.piper.model_path:
                model_path = Path(self.config.piper.model_path)
            else:
                model_files = list(voices_dir.glob("*.onnx"))
                if not model_files:
                    self.logger.log_warning(f"No Piper voice model found in {voices_dir}")
                    return False
                model_path = model_files[0]
            
            config_path = self.config.piper.config_path
            if not config_path:
                config_path = model_path.with_suffix(".json")
            
            if not model_path.exists():
                self.logger.log_error(f"Piper model not found: {model_path}")
                return False
            
            if not Path(config_path).exists():
                self.logger.log_error(f"Piper config not found: {config_path}")
                return False
            
            self.logger.log_info(f"Loading Piper voice: {model_path}")
            self.voice = PiperVoice.load(
                str(model_path),
                str(config_path),
                use_cuda=self.config.piper.use_cuda
            )
            
            self.logger.log_info("Piper voice loaded successfully")
            return True
            
        except Exception as e:
            self.logger.log_error(f"Failed to load Piper voice: {e}", exc_info=True)
            return False
    
    def speak(self, text: str, output_path: Optional[str] = None) -> bool:
        if self.voice is None:
            if not self.load_voice():
                return False
        
        if not text or not text.strip():
            self.logger.log_warning("Empty text for TTS")
            return False
        
        try:
            if output_path:
                with wave.open(output_path, "wb") as wav_file:
                    self.voice.synthesize(text, wav_file)
                self.logger.log_info(f"Speech saved to {output_path}")
            else:
                self.logger.log_info(f"Speaking: {text}")
            
            return True
            
        except Exception as e:
            self.logger.log_error(f"TTS error: {e}", exc_info=True)
            return False
    
    def is_available(self) -> bool:
        return PIPER_AVAILABLE and PiperVoice is not None


