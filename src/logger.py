import logging
import sys
from pathlib import Path

def get_logger(name:str, log_file: str=None, level:int=logging.INFO)->logging.Logger:
    """
    Returns a configured logger with a consistent format.
    
    Args:
        name: logger name (usually __name__)
        log_file: if provided, logs will also be written to this file
        level: logging level (e.g., logging.INFO, logging.DEBUG)
    
    Returns:
        logging.Logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    #avoid adding duplicate handlers if logger already exists
    if logger.handlers:
        return logger
    
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    #console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    #file handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger

