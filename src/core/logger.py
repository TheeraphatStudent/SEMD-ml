import logging
import sys
from typing import Optional


def setup_logging(log_level: str = 'INFO', log_format: Optional[str] = None):
    if log_format is None:
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
