from typing import Callable

from inference_gym import using_jax as gym

from .spec import TargetSpec


def gym_vector_log_prob(target: gym.targets.VectorModel) -> Callable:
    """Unnormalized log density on the unconstrained (R^d) parameterization."""

    def log_prob(x):
        y = target.default_event_space_bijector(x)
        fldj = target.default_event_space_bijector.forward_log_det_jacobian(x)
        return target.unnormalized_log_prob(y) + fldj

    return log_prob


def ill_conditioned_gaussian(*, ndims: int, seed: int = 0) -> TargetSpec:
    target = gym.targets.VectorModel(
        gym.targets.IllConditionedGaussian(ndims=ndims, seed=seed),
        flatten_sample_transformations=True,
    )
    return TargetSpec(
        log_prob=gym_vector_log_prob(target),
        dim=ndims,
        name=target.name,
    )


def banana(*, curvature: float = 0.05) -> TargetSpec:
    target = gym.targets.VectorModel(
        gym.targets.Banana(curvature=curvature),
        flatten_sample_transformations=True,
    )
    return TargetSpec(
        log_prob=gym_vector_log_prob(target),
        dim=2,
        name=target.name,
    )
