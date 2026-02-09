from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    root: Path = Path(__file__).resolve().parents[2]
    data: Path = root / "data"
    raw: Path = data / "raw"
    processed: Path = data / "processed"
    models: Path = root / "models"


PATHS = Paths()
