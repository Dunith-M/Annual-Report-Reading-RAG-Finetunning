from loguru import logger
from pathlib import Path

LOG_DIR = Path("artifacts/reports")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logger.add(
    LOG_DIR / "ledger_mind.log",
    rotation="1 MB",
    retention="7 days",
    level="INFO",
)

def get_logger():
    return logger