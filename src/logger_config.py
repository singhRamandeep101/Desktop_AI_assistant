import logging
import sys
from pathlib import Path
from typing import Optional


class FridayLogger:
    def __init__(self, log_dir: str = "logs", log_level: str = "INFO"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.main_logger = self._setup_logger("friday_main", "main.log", log_level)
        self.error_logger = self._setup_logger("friday_errors", "errors.log", "ERROR")
        self.performance_logger = self._setup_logger("friday_performance", "performance.log", "INFO")
    
    def _setup_logger(
        self,
        name: str,
        filename: str,
        level: str,
        console_level: Optional[str] = None
    ) -> logging.Logger:
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.upper()))
        
        logger.handlers.clear()
        
        file_handler = logging.FileHandler(
            self.log_dir / filename,
            encoding="utf-8"
        )
        file_handler.setLevel(getattr(logging, level.upper()))
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        if console_level is None:
            console_level = "WARNING"
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, console_level.upper()))
        console_formatter = logging.Formatter("%(levelname)s - %(message)s")
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        logger.propagate = False
        
        return logger
    
    def log_info(self, message: str, logger_name: str = "main"):
        logger = getattr(self, f"{logger_name}_logger", self.main_logger)
        logger.info(message)
    
    def log_error(self, message: str, exc_info: bool = False):
        self.error_logger.error(message, exc_info=exc_info)
        self.main_logger.error(message)
    
    def log_warning(self, message: str):
        self.main_logger.warning(message)
    
    def log_debug(self, message: str):
        self.main_logger.debug(message)
    
    def log_performance(self, operation: str, duration: float):
        self.performance_logger.info(f"{operation}: {duration:.3f}s")
    
    def set_level(self, level: str, logger_name: str = "main"):
        logger = getattr(self, f"{logger_name}_logger", self.main_logger)
        logger.setLevel(getattr(logging, level.upper()))
        
        for handler in logger.handlers:
            handler.setLevel(getattr(logging, level.upper()))


