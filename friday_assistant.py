#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.audio_handler import AudioHandler
from src.command_parser import CommandParser
from src.config import AssistantConfig
from src.logger_config import FridayLogger
from src.transcription_service import TranscriptionService
from src.tts_service import TTSService


class FridayAssistant:
    def __init__(self, config: AssistantConfig):
        self.config = config
        self.logger = FridayLogger(config.logging.log_dir, config.logging.log_level)
        self.audio_handler = AudioHandler(config, self.logger)
        self.transcription_service = TranscriptionService(config, self.logger)
        self.tts_service = TTSService(config, self.logger)
        self.command_parser = CommandParser(self.logger)
    
    def initialize(self) -> bool:
        self.logger.log_info("Initializing Friday assistant...")
        
        if not self.transcription_service.load_model():
            self.logger.log_error("Failed to initialize transcription service")
            return False
        
        if self.tts_service.is_available():
            self.tts_service.load_voice()
        else:
            self.logger.log_warning("TTS service not available - responses will be text-only")
        
        self.logger.log_info("Friday assistant initialized successfully")
        return True
    
    def process_command(self, audio=None):
        if audio is None:
            audio = self.audio_handler.record_audio()
            if audio is None:
                return None
        
        audio = self.audio_handler.preprocess_audio(audio)
        
        if not self.audio_handler.validate_audio(audio):
            return None
        
        text = self.transcription_service.transcribe_command(audio)
        if not text:
            return None
        
        self.logger.log_info("Command received")
        command = self.command_parser.parse_command(text)
        
        return command
    
    def run_interactive(self):
        if not self.initialize():
            self.logger.log_error("Failed to initialize assistant")
            return
        
        self.logger.log_info("Friday assistant ready. Say 'exit' to quit.")
        
        try:
            while True:
                command = self.process_command()
                if command:
                    if command["intent"] == "exit":
                        self.logger.log_info("Exiting...")
                        break
                    self._handle_command(command)
                else:
                    self.logger.log_warning("Failed to process command")
        except KeyboardInterrupt:
            self.logger.log_info("Interrupted by user")
        finally:
            self.cleanup()
    
    def _handle_command(self, command: dict):
        intent = command.get("intent", "unknown")
        text = command.get("text", "")
        
        self.logger.log_info(f"Handling command: {intent} - {text}")
        
        response = self.command_parser.get_response(command)
        
        if self.tts_service.is_available():
            self.tts_service.speak(response)
        else:
            print(f"Friday: {response}")
    
    def cleanup(self):
        self.audio_handler.cleanup()
        self.logger.log_info("Cleanup completed")


def main():
    parser = argparse.ArgumentParser(description="Friday Voice Assistant")
    parser.add_argument("--config", type=str, help="Path to configuration YAML/JSON file")
    parser.add_argument("--whisper-model", type=str, help="Whisper model to use (overrides config)")
    parser.add_argument("--log-level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level (overrides config)")
    parser.add_argument("--no-tts", action="store_true", help="Disable text-to-speech")
    
    args = parser.parse_args()
    
    try:
        if args.config:
            config = AssistantConfig.from_file(args.config)
        else:
            default_config = Path("config.yaml")
            if default_config.exists():
                config = AssistantConfig.from_file(default_config)
            else:
                config = AssistantConfig.load()
    except Exception as e:
        print(f"Error loading config: {e}")
        print("Using default configuration...")
        config = AssistantConfig.load()
    
    if args.whisper_model:
        config.whisper.model = args.whisper_model
    if args.log_level:
        config.logging.log_level = args.log_level
    if args.no_tts:
        config.piper.model_path = ""
    
    assistant = FridayAssistant(config)
    assistant.run_interactive()


if __name__ == "__main__":
    main()

