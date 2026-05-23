import json
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> None:
    """
    Create directory if it does not exist.
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(data: Any, output_path: str | Path) -> None:
    """
    Save Python object as pretty JSON.
    """
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(input_path: str | Path) -> Any:
    """
    Load JSON file.
    """
    input_path = Path(input_path)

    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)