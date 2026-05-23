from loguru import logger
from pathlib import Path
import sys


def setup_logger(log_file: str = "artifacts/reports/task2_pdf_chunking.log"):
    """
    Configure project logger.

    Logs are written to:
    artifacts/reports/task2_pdf_chunking.log
    """

    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger.remove()

    logger.add(
        sys.stdout,
        level="INFO",
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
               "<level>{message}</level>",
    )

    logger.add(
        log_path,
        level="DEBUG",
        rotation="1 MB",
        retention="7 days",
        compression="zip",
    )

    return logger