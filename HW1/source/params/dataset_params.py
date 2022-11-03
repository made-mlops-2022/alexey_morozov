from dataclasses import dataclass
from typing import List


@dataclass
class DataParams:
    data_path: str
    val_data_path: str
    numeric_features: List[str]
    categorical_features: List[str]
    target: str
