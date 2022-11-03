from dataclasses import dataclass, field
from typing import Union
from xml.dom import NotFoundErr


@dataclass
class LRParams:
    penalty: str = field(default="l2")
    C: float = field(default=1.0)
    solver: str = field(default="lbfgs")
    random_state: int = field(default=17)
    verbose: int = field(default=0)
    n_jobs: int = field(default=-1)


@dataclass
class RFParams:
    n_estimators: int = field(default=100)
    criterion: str = field(default="gini")
    max_depth: Union[None, int] = field(default=None)
    min_samples_split: Union[float, int] = field(default=2)
    min_samples_leaf: Union[float, int] = field(default=1)
    random_state: int = field(default=17)
    verbose: int = field(default=0)
    n_jobs: int = field(default=-1)


model_params = Union[LRParams, RFParams]


@dataclass
class ModelParams:
    model: str
    model_path: str
    preprocessor_path: str
    params: model_params = field(init=False)

    def __post_init__(self):
        if self.model == "lr":
            self.params = LRParams()
        elif self.model == "rf":
            self.params = RFParams()
        else:
            raise NotFoundErr("Wrong model name")
