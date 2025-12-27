#!/usr/bin/env python3
"""
Friday Voice Assistant - Improved Version
A comprehensive voice assistant with transcription, command parsing, and TTS capabilities.
"""

import argparse
import json
import logging
import os
import sys
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import onnxruntime
from faster_whisper import WhisperModel

# Try to import piper - handle both installed package and local path
PIPER_AVAILABLE = False
PiperVoice = None
PhonemeType = None
PiperConfig = None
BOS = EOS = PAD = None
audio_float_to_int16 = None

try:
    from piper.voice import PiperVoice
    from piper.config import PhonemeType, PiperConfig
    from piper.const import BOS, EOS, PAD
    from piper.util import audio_float_to_int16
    PIPER_AVAILABLE = True
except ImportError:
    # Try local piper path
    piper_path = Path(__file__).parent / "piper-master" / "src" / "python_run"
    if piper_path.exists():
        sys.path.insert(0, str(piper_path))
        try:
            from piper.voice import PiperVoice
            from piper.config import PhonemeType, PiperConfig
            from piper.const import BOS, EOS, PAD
            from piper.util import audio_float_to_int16
            PIPER_AVAILABLE = True
        except ImportError:
            pass

# Try to import audio recording libraries
try:
    import pyaudio
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("Warning: pyaudio not available. Audio recording will be disabled.")


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class AssistantConfig:
    """Configuration for the Friday assistant."""
    # Whisper model settings
    whisper_model: str = "base"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    whisper_beam_size: int = 1
    whisper_vad_filter: bool = True
    
    # Piper TTS settings
    piper_model_path: Optional[str] = None
    piper_config_path: Optional[str] = None
    piper_use_cuda: bool = False
    
    # Audio settings
    sample_rate: int = 16000
    chunk_size: int = 1024
    channels: int = 1
    audio_format: int = None  # Will be set to pyaudio.paInt16 if available
    
    # Performance settings
    enable_model_caching: bool = True
    transcription_timeout: float = 30.0
    
    # Error handling
    max_retries: int = 3
    retry_delay: float = 0.5
    
    # Logging
    log_dir: str = "logs"
    log_level: str = "INFO"
    
    # Paths
    piper_voices_dir: str = "piper_voices"
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "AssistantConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})
    
    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> "AssistantConfig":
        """Load config from JSON file."""
        with open(config_path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


# ============================================================================
# Logging Setup
# ============================================================================

class FridayLogger:
    """Centralized logging for Friday assistant."""
    
    def __init__(self, log_dir: str = "logs", log_level: str = "INFO"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup loggers
        self.main_logger = self._setup_logger("friday_main", "main.log", log_level)
        self.error_logger = self._setup_logger("friday_errors", "errors.log", "ERROR")
        self.performance_logger = self._setup_logger("friday_performance", "performance.log", "INFO")
    
    def _setup_logger(self, name: str, filename: str, level: str) -> logging.Logger:
        """Setup a logger with file and console handlers."""
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.upper()))
        
        # Remove existing handlers
        logger.handlers.clear()
        
        # File handler
        file_handler = logging.FileHandler(self.log_dir / filename, encoding="utf-8")
        file_handler.setLevel(getattr(logging, level.upper()))
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.WARNING)  # Only show warnings/errors on console
        console_formatter = logging.Formatter("%(levelname)s - %(message)s")
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    def log_info(self, message: str, logger_name: str = "main"):
        """Log info message."""
        logger = getattr(self, f"{logger_name}_logger", self.main_logger)
        logger.info(message)
    
    def log_error(self, message: str, exc_info: bool = False):
        """Log error message."""
        self.error_logger.error(message, exc_info=exc_info)
        self.main_logger.error(message)
    
    def log_warning(self, message: str):
        """Log warning message."""
        self.main_logger.warning(message)
    
    def log_performance(self, operation: str, duration: float):
        """Log performance metric."""
        self.performance_logger.info(f"{operation}: {duration:.3f}s")


# ============================================================================
# Audio Handler
# ============================================================================

class AudioHandler:
    """Handles audio recording and preprocessing."""
    
    def __init__(self, config: AssistantConfig, logger: FridayLogger):
        self.config = config
        self.logger = logger
        self.audio = None
        self.stream = None
        
        if AUDIO_AVAILABLE:
            self.audio = pyaudio.PyAudio()
    
    def record_audio(self, duration: float = 3.0) -> Optional[np.ndarray]:
        """Record audio for specified duration."""
        if not AUDIO_AVAILABLE or not self.audio:
            self.logger.log_error("Audio recording not available")
            return None
        
        try:
            self.logger.log_info(f"Recording audio for {duration}s")
            
            # Set audio format if not already set
            audio_format = self.config.audio_format
            if audio_format is None and AUDIO_AVAILABLE:
                audio_format = pyaudio.paInt16
            
            self.stream = self.audio.open(
                format=audio_format,
                channels=self.config.channels,
                rate=self.config.sample_rate,
                input=True,
                frames_per_buffer=self.config.chunk_size
            )
            
            frames = []
            num_chunks = int(self.config.sample_rate / self.config.chunk_size * duration)
            
            for _ in range(num_chunks):
                data = self.stream.read(self.config.chunk_size)
                frames.append(data)
            
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
            
            # Convert to numpy array
            audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
            # Normalize to float32 [-1.0, 1.0]
            audio_data = audio_data.astype(np.float32) / 32768.0
            
            # Calculate audio statistics
            max_amp = np.max(np.abs(audio_data)) if len(audio_data) > 0 else 0
            rms = np.sqrt(np.mean(audio_data**2)) if len(audio_data) > 0 else 0
            
            self.logger.log_info(f"Recorded {len(audio_data)} samples")
            self.logger.log_info(f"Audio stats - Max amplitude: {max_amp:.6f}, RMS: {rms:.6f}")
            
            return audio_data
            
        except Exception as e:
            self.logger.log_error(f"Error recording audio: {e}", exc_info=True)
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None
            return None
    
    def preprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        """Preprocess audio for transcription."""
        # Ensure float32 dtype (fixes ONNX error)
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Normalize if needed
        max_val = np.max(np.abs(audio)) if len(audio) > 0 else 1.0
        if max_val > 1.0:
            audio = audio / max_val
        elif max_val == 0.0:
            self.logger.log_warning("Audio is completely silent (all zeros)")
        
        return audio
    
    def cleanup(self):
        """Cleanup audio resources."""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        
        if self.audio:
            self.audio.terminate()
            self.audio = None


# ============================================================================
# Transcription Service
# ============================================================================

class TranscriptionService:
    """Handles audio transcription with error recovery."""
    
    def __init__(self, config: AssistantConfig, logger: FridayLogger):
        self.config = config
        self.logger = logger
        self.model: Optional[WhisperModel] = None
        self._model_cache = {}
    
    def load_model(self) -> bool:
        """Load Whisper model with caching."""
        if self.config.enable_model_caching and self.model is not None:
            self.logger.log_info("Using cached Whisper model")
            return True
        
        try:
            self.logger.log_info(f"Loading Whisper model: {self.config.whisper_model}")
            start_time = time.time()
            
            self.model = WhisperModel(
                self.config.whisper_model,
                device=self.config.whisper_device,
                compute_type=self.config.whisper_compute_type
            )
            
            load_time = time.time() - start_time
            self.logger.log_info(f"Model loaded in {load_time:.2f}s")
            self.logger.log_performance("model_load", load_time)
            
            return True
            
        except Exception as e:
            self.logger.log_error(f"Failed to load Whisper model: {e}", exc_info=True)
            return False
    
    def transcribe_audio(
        self,
        audio: np.ndarray,
        language: Optional[str] = None,
        task: str = "transcribe"
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        """
        Transcribe audio with retry logic and error handling.
        
        Returns:
            Tuple of (transcribed_text, info_dict)
        """
        if self.model is None:
            if not self.load_model():
                return None, {}
        
        # Preprocess audio to ensure float32 (fixes ONNX error)
        audio = self._ensure_float32(audio)
        
        self.logger.log_info("Starting audio transcription")
        start_time = time.time()
        
        for attempt in range(self.config.max_retries):
            try:
                # Transcription with optimized parameters
                segments, info = self.model.transcribe(
                    audio,
                    beam_size=self.config.whisper_beam_size,
                    vad_filter=self.config.whisper_vad_filter,
                    language=language,
                    task=task
                )
                
                # Extract text from segments
                text_segments = [segment.text.strip() for segment in segments]
                transcribed_text = " ".join(text_segments).strip()
                
                transcription_time = time.time() - start_time
                self.logger.log_info("Transcription completed")
                self.logger.log_performance("transcribe_audio", transcription_time)
                
                if not transcribed_text:
                    self.logger.log_warning("Empty transcription result")
                    return None, info
                
                return transcribed_text, info
                
            except onnxruntime.capi.onnxruntime_pybind11_state.InvalidArgument as e:
                error_msg = str(e)
                if "data type" in error_msg.lower() or "tensor" in error_msg.lower():
                    # ONNX dtype error - fix and retry
                    self.logger.log_warning(f"ONNX dtype error (attempt {attempt + 1}/{self.config.max_retries}): {error_msg}")
                    audio = self._ensure_float32(audio)
                    if attempt < self.config.max_retries - 1:
                        time.sleep(self.config.retry_delay * (attempt + 1))
                        continue
                
                self.logger.log_error(f"Audio transcription: {error_msg}", exc_info=True)
                return None, {}
                
            except Exception as e:
                self.logger.log_error(f"Transcription error: {e}", exc_info=True)
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))
                    continue
                return None, {}
        
        return None, {}
    
    def _ensure_float32(self, audio: np.ndarray) -> np.ndarray:
        """Ensure audio is float32 dtype (fixes ONNX errors)."""
        if audio.dtype != np.float32:
            self.logger.log_info(f"Converting audio from {audio.dtype} to float32")
            audio = audio.astype(np.float32)
        return audio
    
    def transcribe_command(self, audio: np.ndarray, vad_filter: Optional[bool] = None) -> Optional[str]:
        """Transcribe audio and return command text."""
        # Allow VAD filter override
        original_vad = self.config.whisper_vad_filter
        if vad_filter is not None:
            self.config.whisper_vad_filter = vad_filter
        
        try:
            text, _ = self.transcribe_audio(audio, task="transcribe")
            return text
        finally:
            self.config.whisper_vad_filter = original_vad


# ============================================================================
# TTS Service (Piper)
# ============================================================================

class TTSService:
    """Handles text-to-speech using Piper."""
    
    def __init__(self, config: AssistantConfig, logger: FridayLogger):
        self.config = config
        self.logger = logger
        self.voice: Optional[Any] = None
        
        if PiperVoice is None:
            self.logger.log_warning("Piper TTS not available - TTS will be disabled")
    
    def load_voice(self) -> bool:
        """Load Piper voice model."""
        if not PIPER_AVAILABLE or PiperVoice is None:
            self.logger.log_warning("Piper TTS not available - install piper-phonemize and onnxruntime")
            self.logger.log_warning("TTS will be disabled. Assistant will work but won't speak responses.")
            return False
            
        try:
            # Try to find voice model
            voices_dir = Path(self.config.piper_voices_dir)
            
            if not voices_dir.exists():
                self.logger.log_warning(f"Piper voices directory not found: {voices_dir}")
                self.logger.log_warning("Download voice models from: https://github.com/rhasspy/piper/releases")
                return False
            
            if self.config.piper_model_path:
                model_path = Path(self.config.piper_model_path)
            else:
                # Look for default voice
                model_files = list(voices_dir.glob("*.onnx"))
                if not model_files:
                    self.logger.log_warning(f"No Piper voice model found in {voices_dir}")
                    self.logger.log_warning("Download voice models from: https://github.com/rhasspy/piper/releases")
                    return False
                model_path = model_files[0]
            
            config_path = self.config.piper_config_path
            if not config_path:
                config_path = model_path.with_suffix(".json")
            
            if not model_path.exists():
                self.logger.log_error(f"Piper model not found: {model_path}")
                return False
            
            if not Path(config_path).exists():
                self.logger.log_error(f"Piper config not found: {config_path}")
                return False
            
            self.logger.log_info(f"Loading Piper voice: {model_path}")
            if PiperVoice:
                self.voice = PiperVoice.load(
                    str(model_path),
                    str(config_path),
                    use_cuda=self.config.piper_use_cuda
                )
            else:
                return False
            
            self.logger.log_info("Piper voice loaded successfully")
            return True
            
        except Exception as e:
            self.logger.log_error(f"Failed to load Piper voice: {e}", exc_info=True)
            return False
    
    def speak(self, text: str, output_path: Optional[str] = None) -> bool:
        """Synthesize and optionally save speech."""
        if self.voice is None:
            if not self.load_voice():
                return False
        
        try:
            if output_path:
                with wave.open(output_path, "wb") as wav_file:
                    self.voice.synthesize(text, wav_file)
                self.logger.log_info(f"Speech saved to {output_path}")
            else:
                # Play audio directly (requires additional audio playback library)
                # For now, just log
                self.logger.log_info(f"Speaking: {text}")
            
            return True
            
        except Exception as e:
            self.logger.log_error(f"TTS error: {e}", exc_info=True)
            return False


# ============================================================================
# Command Parser
# ============================================================================

class CommandParser:
    """Parses transcribed commands and extracts intents."""
    
    def __init__(self, logger: FridayLogger):
        self.logger = logger
    
    def parse_command(self, text: str) -> Dict[str, Any]:
        """
        Parse command text and extract intent.
        
        Returns:
            Dictionary with 'intent', 'action', 'parameters', etc.
        """
        if not text:
            return {"intent": "unknown", "text": ""}
        
        text_lower = text.lower().strip()
        start_time = time.time()
        
        # Simple command parsing (can be enhanced with NLP)
        intent = "unknown"
        action = None
        parameters = {}
        
        # Example command patterns
        if any(word in text_lower for word in ["hello", "hi", "hey"]):
            intent = "greeting"
        elif any(word in text_lower for word in ["time", "what time"]):
            intent = "time_query"
        elif any(word in text_lower for word in ["date", "what date"]):
            intent = "date_query"
        elif any(word in text_lower for word in ["weather", "temperature"]):
            intent = "weather_query"
        elif any(word in text_lower for word in ["stop", "exit", "quit"]):
            intent = "exit"
        else:
            intent = "general"
        
        parse_time = time.time() - start_time
        self.logger.log_performance("parse_command", parse_time)
        
        return {
            "intent": intent,
            "action": action,
            "text": text,
            "parameters": parameters
        }


# ============================================================================
# Main Assistant Class
# ============================================================================

class FridayAssistant:
    """Main Friday voice assistant."""
    
    def __init__(self, config: Optional[AssistantConfig] = None):
        self.config = config or AssistantConfig()
        self.logger = FridayLogger(self.config.log_dir, self.config.log_level)
        self.audio_handler = AudioHandler(self.config, self.logger)
        self.transcription_service = TranscriptionService(self.config, self.logger)
        self.tts_service = TTSService(self.config, self.logger)
        self.command_parser = CommandParser(self.logger)
        
        self.logger.log_info("setup_logging:91 - Logging system initialized")
    
    def initialize(self) -> bool:
        """Initialize all services."""
        self.logger.log_info("Initializing Friday assistant...")
        
        # Load models
        if not self.transcription_service.load_model():
            self.logger.log_error("Failed to initialize transcription service")
            return False
        
        # TTS is optional
        self.tts_service.load_voice()
        
        self.logger.log_info("Friday assistant initialized successfully")
        return True
    
    def process_command(self, audio: Optional[np.ndarray] = None) -> Optional[Dict[str, Any]]:
        """Process a voice command."""
        # Record audio if not provided
        if audio is None:
            audio = self.audio_handler.record_audio()
            if audio is None:
                self.logger.log_warning("Failed to record audio")
                return None
        
        # Preprocess audio
        audio = self.audio_handler.preprocess_audio(audio)
        
        # Validate audio before transcription
        max_val = np.max(np.abs(audio)) if len(audio) > 0 else 0
        min_threshold = 0.0005  # Very low threshold - allow quiet audio through
        
        if max_val < min_threshold:
            self.logger.log_warning(f"Audio appears to be silent (max amplitude: {max_val:.6f})")
            self.logger.log_warning("Check microphone: ensure it's not muted and volume is adequate")
            return None
        elif max_val < 0.005:
            # Audio is very quiet - will attempt but may not work well
            self.logger.log_warning(f"Audio very quiet (max amplitude: {max_val:.6f}) - transcription may fail")
            self.logger.log_warning("Try: Speak louder, increase mic volume, or move closer to microphone")
        elif max_val < 0.01:
            # Audio is quiet but should work
            self.logger.log_info(f"Audio detected but quiet (max amplitude: {max_val:.6f}) - attempting transcription")
        else:
            self.logger.log_info(f"Audio validated: {len(audio)} samples, max amplitude: {max_val:.4f}")
        
        # Transcribe - disable VAD for very quiet audio as it may filter out speech
        use_vad = self.config.whisper_vad_filter
        if max_val < 0.01:
            # For very quiet audio, VAD might filter out the speech
            use_vad = False
            self.logger.log_info("Disabling VAD filter for quiet audio to improve transcription")
        
        text = self.transcription_service.transcribe_command(audio, vad_filter=use_vad)
        
        if not text:
            self.logger.log_warning("Transcription returned empty result - no speech detected")
            self.logger.log_warning("Possible causes: Audio too quiet, no speech in recording, or microphone issues")
            return None
        
        self.logger.log_info("Command received")
        
        # Parse command
        command = self.command_parser.parse_command(text)
        
        return command
    
    def run_interactive(self):
        """Run assistant in interactive mode."""
        if not self.initialize():
            self.logger.log_error("Failed to initialize assistant")
            return
        
        # Check audio availability
        if not AUDIO_AVAILABLE:
            self.logger.log_error("Audio recording not available - install pyaudio")
            self.logger.log_error("Windows: pip install pyaudio")
            return
        
        # Test microphone
        self.logger.log_info("Testing microphone (speak now for 1 second)...")
        test_audio = self.audio_handler.record_audio(duration=1.0)
        if test_audio is not None:
            max_amp = np.max(np.abs(test_audio)) if len(test_audio) > 0 else 0
            if max_amp > 0.01:
                self.logger.log_info(f"✓ Microphone test passed - Good audio level: {max_amp:.6f}")
            elif max_amp > 0.001:
                self.logger.log_warning(f"⚠ Microphone test: Low audio level ({max_amp:.6f})")
                self.logger.log_warning("   Tip: Increase microphone volume in Windows Settings")
                self.logger.log_warning("   Or speak louder/closer to microphone for better results")
            else:
                self.logger.log_warning(f"✗ Microphone test: Very quiet audio ({max_amp:.6f})")
                self.logger.log_warning("   Check: Microphone not muted, volume adequate, correct input device")
        else:
            self.logger.log_warning("✗ Microphone test failed - check microphone connection")
        
        self.logger.log_info("Friday assistant ready. Say 'exit' to quit.")
        self.logger.log_info("Speak clearly into your microphone...")
        
        try:
            while True:
                command = self.process_command()
                if command:
                    if command["intent"] == "exit":
                        self.logger.log_info("Exiting...")
                        break
                    
                    # Handle command (placeholder)
                    self._handle_command(command)
                else:
                    self.logger.log_warning("No command detected. Try speaking louder or closer to microphone.")
        
        except KeyboardInterrupt:
            self.logger.log_info("Interrupted by user")
        finally:
            self.cleanup()
    
    def _handle_command(self, command: Dict[str, Any]):
        """Handle parsed command."""
        intent = command.get("intent", "unknown")
        text = command.get("text", "")
        
        self.logger.log_info(f"Handling command: {intent} - {text}")
        
        # Placeholder for command handling logic
        if intent == "greeting":
            self.tts_service.speak("Hello! How can I help you?")
        elif intent == "time_query":
            import datetime
            current_time = datetime.datetime.now().strftime("%I:%M %p")
            self.tts_service.speak(f"The current time is {current_time}")
        elif intent == "date_query":
            import datetime
            current_date = datetime.datetime.now().strftime("%B %d, %Y")
            self.tts_service.speak(f"Today is {current_date}")
        else:
            self.tts_service.speak(f"I heard: {text}")
    
    def cleanup(self):
        """Cleanup resources."""
        self.audio_handler.cleanup()
        self.logger.log_info("Cleanup completed")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Friday Voice Assistant")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration JSON file"
    )
    parser.add_argument(
        "--whisper-model",
        type=str,
        default="base",
        help="Whisper model to use (default: base)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Load config
    if args.config and Path(args.config).exists():
        config = AssistantConfig.from_file(args.config)
    else:
        config = AssistantConfig()
    
    # Override with command line args
    if args.whisper_model:
        config.whisper_model = args.whisper_model
    if args.log_level:
        config.log_level = args.log_level
    
    # Create and run assistant
    assistant = FridayAssistant(config)
    assistant.run_interactive()


if __name__ == "__main__":
    main()

