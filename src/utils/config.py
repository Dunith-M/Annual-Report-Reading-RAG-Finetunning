from pathlib import Path
import yaml
from pydantic import BaseModel


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ProjectPaths(BaseModel):
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    artifacts_dir: str = "artifacts"


def resolve_project_path(path: str | Path) -> Path:
    """
    Resolve repo-relative paths from anywhere, including notebooks.
    """
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def read_yaml(path: str | Path) -> dict:
    path = resolve_project_path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)
