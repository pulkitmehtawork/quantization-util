from abc import ABC, abstractmethod
from pathlib import Path


class BaseQuantizer(ABC):
    """Abstract base for all quantization backends."""

    def __init__(self, model_name_or_path: str):
        self.model_name_or_path = model_name_or_path

    @abstractmethod
    def quantize(self, output_dir: str, **kwargs) -> Path:
        """Run quantization and return path to output artifact."""

    def _ensure_output_dir(self, output_dir: str) -> Path:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path
