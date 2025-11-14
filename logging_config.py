import logging
import os
from datetime import datetime


def get_logger(name,log_dir='./logs'):
    os.makedirs(log_dir,exist_ok=True)
    log_file=os.path.join(log_dir,f"{name}_{datetime.now():%Y%m%d_%H%M%S}.log")

    logger=logging.getLogger(name)
    logger.setLevel(logging.INFO)

    file_handler=logging.FileHandler(log_file)
    stream_handler=logging.StreamHandler()

    formatter=logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    return logger
