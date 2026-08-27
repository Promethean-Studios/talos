"""Download + prepare scripts for legally-usable public datasets."""
from __future__ import annotations

from data.setup.datasets import DATASETS, DatasetSpec, get_dataset, known_datasets

__all__ = ["DATASETS", "DatasetSpec", "get_dataset", "known_datasets"]
