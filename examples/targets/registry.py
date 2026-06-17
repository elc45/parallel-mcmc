from typing import Callable

from .blr_german_credit import blr_german_credit
from .gaussian import gaussian_2d, gaussian_10d
from .gym import banana, ill_conditioned_gaussian
from .spec import TargetSpec

TargetFactory = Callable[..., TargetSpec]

TARGETS: dict[str, TargetFactory] = {
    "ill_conditioned_gaussian": ill_conditioned_gaussian,
    "banana": banana,
    "blr_german_credit": blr_german_credit,
    "gaussian_2d": gaussian_2d,
    "gaussian_10d": gaussian_10d,
}


def load_target(name: str, params: dict | None = None) -> TargetSpec:
    if name not in TARGETS:
        raise ValueError(f"Unknown target {name!r}. Choose from {sorted(TARGETS)}")
    return TARGETS[name](**(params or {}))
