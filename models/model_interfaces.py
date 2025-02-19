from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

class BaseRetriever(ABC):
    """Abstract base class for all image retrieval models."""

    def __init__(self, image_dir: str):
        self.image_dir = Path(image_dir)
        self.image_paths = list(self.image_dir.glob("*.jpg")) + list(self.image_dir.glob("*.png"))
        self.preprocessed_data: Dict[str, Any] = {}
        self.is_initialized = False

    @abstractmethod
    def preprocess(self) -> None:
        """Preprocess and index all images in the directory."""
        pass

    @abstractmethod
    def retrieve(self, query: str, n: int = 5) -> List[str]:
        """Retrieve n most relevant images for the given query."""
        pass

    def initialize(self) -> None:
        """Initialize the model and preprocess images if not already done."""
        if not self.is_initialized:
            self.preprocess()
            self.is_initialized = True
