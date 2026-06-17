from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class TargetSpec:
    log_prob: Callable
    dim: int
    name: str
