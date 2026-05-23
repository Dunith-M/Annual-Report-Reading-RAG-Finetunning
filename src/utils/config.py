from pathlib import Path
import yaml
from pydantic import BaseModel


class ProjectPaths(BaseModel):
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    artifacts_dir: str = "artifacts"


def read_yaml(path: str | Path) -> dict:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)