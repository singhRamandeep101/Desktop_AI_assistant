from typing import Optional

import numpy as np

try:
    import pyaudio
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    pyaudio = None


class AudioHandler:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.audio = None
        self.stream = None
        
        if AUDIO_AVAILABLE:
            try:
                self.audio = pyaudio.PyAudio()
            except Exception as e:
                self.logger.log_warning(f"Failed to initialize PyAudio: {e}")
                self.audio = None
        else:
            self.logger.log_warning("PyAudio not available - audio recording disabled")
    
    def record_audio(self, duration: float = 3.0) -> Optional[np.ndarray]:
        if not AUDIO_AVAILABLE or not self.audio:
            self.logger.log_error("Audio recording not available")
            return None
        
        try:
            self.logger.log_info(f"Recording audio for {duration}s")
            
            if self.config.audio.format == "int16":
                audio_format = pyaudio.paInt16
                dtype = np.int16
            else:
                audio_format = pyaudio.paInt16
                dtype = np.int16
            
            self.stream = self.audio.open(
                format=audio_format,
                channels=self.config.audio.channels,
                rate=self.config.audio.sample_rate,
                input=True,
                frames_per_buffer=self.config.audio.chunk_size
            )
            
            frames = []
            num_chunks = int(self.config.audio.sample_rate / self.config.audio.chunk_size * duration)
            
            for _ in range(num_chunks):
                data = self.stream.read(self.config.audio.chunk_size, exception_on_overflow=False)
                frames.append(data)
            
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
            
            audio_data = np.frombuffer(b''.join(frames), dtype=dtype)
            audio_data = audio_data.astype(np.float32) / 32768.0
            
            self.logger.log_info(f"Recorded {len(audio_data)} samples")
            return audio_data
            
        except Exception as e:
            self.logger.log_error(f"Error recording audio: {e}", exc_info=True)
            if self.stream:
                try:
                    self.stream.stop_stream()
                    self.stream.close()
                except:
                    pass
                self.stream = None
            return None
    
    def preprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        if audio.dtype != np.float32:
            self.logger.log_debug(f"Converting audio from {audio.dtype} to float32")
            audio = audio.astype(np.float32)
        
        max_val = np.max(np.abs(audio))
        if max_val > 1.0:
            audio = audio / max_val
        elif max_val == 0.0:
            self.logger.log_warning("Audio is silent (all zeros)")
        
        return audio
    
    def validate_audio(self, audio: np.ndarray) -> bool:
        if audio is None or len(audio) == 0:
            self.logger.log_warning("Audio is empty")
            return False
        
        max_val = np.max(np.abs(audio))
        if max_val < 0.01:
            self.logger.log_warning("Audio appears to be silent")
            return False
        
        if audio.dtype != np.float32:
            self.logger.log_warning(f"Audio dtype is {audio.dtype}, expected float32")
            return False
        
        return True
    
    def cleanup(self):
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except:
                pass
            self.stream = None
        
        if self.audio:
            try:
                self.audio.terminate()
            except:
                pass
            self.audio = None


