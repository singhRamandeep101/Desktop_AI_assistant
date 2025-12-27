"""Tests for logger configuration."""

import logging
import tempfile
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.logger_config import FridayLogger


class TestFridayLogger:
    """Tests for FridayLogger."""
    
    def test_initialization(self):
        """Test logger initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = FridayLogger(log_dir=temp_dir, log_level="INFO")
            
            assert logger.log_dir == Path(temp_dir)
            assert logger.main_logger is not None
            assert logger.error_logger is not None
            assert logger.performance_logger is not None
    
    def test_log_info(self):
        """Test info logging."""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = FridayLogger(log_dir=temp_dir, log_level="INFO")
            logger.log_info("Test info message")
            
            # Check that log file was created
            log_file = Path(temp_dir) / "main.log"
            assert log_file.exists()
            
            # Check log content
            content = log_file.read_text()
            assert "Test info message" in content
    
    def test_log_error(self):
        """Test error logging."""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = FridayLogger(log_dir=temp_dir, log_level="ERROR")
            logger.log_error("Test error message")
            
            # Check error log file
            error_log = Path(temp_dir) / "errors.log"
            assert error_log.exists()
            
            content = error_log.read_text()
            assert "Test error message" in content
    
    def test_log_warning(self):
        """Test warning logging."""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = FridayLogger(log_dir=temp_dir, log_level="WARNING")
            logger.log_warning("Test warning message")
            
            log_file = Path(temp_dir) / "main.log"
            assert log_file.exists()
            
            content = log_file.read_text()
            assert "Test warning message" in content
    
    def test_log_performance(self):
        """Test performance logging."""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = FridayLogger(log_dir=temp_dir, log_level="INFO")
            logger.log_performance("test_operation", 1.234)
            
            perf_log = Path(temp_dir) / "performance.log"
            assert perf_log.exists()
            
            content = perf_log.read_text()
            assert "test_operation" in content
            assert "1.234" in content
    
    def test_log_debug(self):
        """Test debug logging."""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = FridayLogger(log_dir=temp_dir, log_level="DEBUG")
            logger.log_debug("Test debug message")
            
            log_file = Path(temp_dir) / "main.log"
            assert log_file.exists()
            
            content = log_file.read_text()
            assert "Test debug message" in content
    
    def test_set_level(self):
        """Test setting log level."""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = FridayLogger(log_dir=temp_dir, log_level="INFO")
            logger.set_level("DEBUG", "main")
            
            assert logger.main_logger.level == logging.DEBUG
    
    def test_multiple_loggers(self):
        """Test that multiple loggers work independently."""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger1 = FridayLogger(log_dir=temp_dir, log_level="INFO")
            logger2 = FridayLogger(log_dir=temp_dir, log_level="DEBUG")
            
            logger1.log_info("Logger 1 message")
            logger2.log_info("Logger 2 message")
            
            log_file = Path(temp_dir) / "main.log"
            content = log_file.read_text()
            assert "Logger 1 message" in content
            assert "Logger 2 message" in content


