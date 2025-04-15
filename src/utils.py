import yaml
from loguru import logger
import os


def load_config(path="config/config.yaml"):
    try:
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded config from {path}")
        return config
    except Exception as e:
        logger.exception(f"Config load error: {e}")
        raise


def setup_logging(log_file="logs/app.log"):
    logger.remove()  # Clear default handler
    logger.add(log_file, rotation="1 MB", level="DEBUG")
    logger.add(lambda msg: print(msg, end=""), level="INFO")
    logger.info("Logging configured")
