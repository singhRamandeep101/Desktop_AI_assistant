import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import onnxruntime
from faster_whisper import WhisperModel


class TranscriptionService:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.model: Optional[WhisperModel] = None
        self._model_loaded = False
    
    def load_model(self) -> bool:
        if self.config.performance.enable_model_caching and self.model is not None:
            self.logger.log_info("Using cached Whisper model")
            return True
        
        try:
            self.logger.log_info(f"Loading Whisper model: {self.config.whisper.model}")
            start_time = time.time()
            
            self.model = WhisperModel(
                self.config.whisper.model,
                device=self.config.whisper.device,
                compute_type=self.config.whisper.compute_type
            )
            
            load_time = time.time() - start_time
            self.logger.log_info(f"Model loaded in {load_time:.2f}s")
            self.logger.log_performance("model_load", load_time)
            self._model_loaded = True
            
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
        if self.model is None:
            if not self.load_model():
                return None, {}
        
        audio = self._ensure_float32(audio)
        
        if not self._validate_audio(audio):
            return None, {}
        
        self.logger.log_info("Starting audio transcription")
        start_time = time.time()
        
        for attempt in range(self.config.error_handling.max_retries):
            try:
                segments, info = self.model.transcribe(
                    audio,
                    beam_size=self.config.whisper.beam_size,
                    vad_filter=self.config.whisper.vad_filter,
                    language=language,
                    task=task
                )
                
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
                    self.logger.log_warning(
                        f"ONNX dtype error (attempt {attempt + 1}/{self.config.error_handling.max_retries}): {error_msg}"
                    )
                    audio = self._ensure_float32(audio)
                    if attempt < self.config.error_handling.max_retries - 1:
                        delay = self.config.error_handling.retry_delay * (attempt + 1)
                        time.sleep(delay)
                        continue
                
                self.logger.log_error(f"Audio transcription: {error_msg}", exc_info=True)
                return None, {}
                
            except Exception as e:
                self.logger.log_error(f"Transcription error: {e}", exc_info=True)
                if attempt < self.config.error_handling.max_retries - 1:
                    delay = self.config.error_handling.retry_delay * (attempt + 1)
                    time.sleep(delay)
                    continue
                return None, {}
        
        return None, {}
    
    def transcribe_command(self, audio: np.ndarray) -> Optional[str]:
        text, _ = self.transcribe_audio(audio, task="transcribe")
        return text
    
    def _ensure_float32(self, audio: np.ndarray) -> np.ndarray:
        if audio.dtype != np.float32:
            self.logger.log_info(f"Converting audio from {audio.dtype} to float32")
            audio = audio.astype(np.float32)
        return audio
    
    def _validate_audio(self, audio: np.ndarray) -> bool:
        if audio is None or len(audio) == 0:
            self.logger.log_warning("Audio is empty")
            return False
        
        max_val = np.max(np.abs(audio))
        if max_val < 0.01:
            self.logger.log_warning("Audio appears to be silent")
            return False
        
        return True


