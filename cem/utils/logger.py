"""
CEM Utils: Logger.
Standardized logging for training and inference.
"""

import logging
import sys
from typing import Optional

def setup_logger(name: str = "CEM", log_file: Optional[str] = None, level=logging.INFO) -> logging.Logger:
    """
    Sets up a logger with console and optional file output.
    
    Args:
        name: Name of the logger.
        log_file: If provided, logs will be saved to this file.
        level: Logging level (default: INFO).
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers if setup_logger is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()
        
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File Handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger