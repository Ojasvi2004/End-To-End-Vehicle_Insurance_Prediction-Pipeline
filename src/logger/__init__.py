import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime
from pathlib import Path

try:
    from from_root import from_root
except ImportError:
    def from_root() -> str:
        # Resolve project root as two levels up from src/logger/__init__.py
        return str(Path(__file__).resolve().parents[2])

LOG_DIR='logs'
LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
MAX_LOG_SIZE=5*1024*1024
BACKUP_COUNT=3

log_dir_path=os.path.join(from_root(),LOG_DIR)
os.makedirs(log_dir_path,exist_ok=True)
log_file_path=os.path.join(log_dir_path,LOG_FILE)

def configure_logger():
    """
    Configration of logger in the project
    """
    
    logger=logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    formatter=logging.Formatter("[%(asctime)s] %(name)s - %(levelname)s - %(message)s")
    
    
    file_handler=RotatingFileHandler(log_file_path,maxBytes=MAX_LOG_SIZE,backupCount=BACKUP_COUNT)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    
    console_handler=logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.DEBUG)
    
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    
configure_logger()
