import jax.numpy as jnp

from .spec import TargetSpec


def diagonal_gaussian(*, variances, name: str) -> TargetSpec:
    """Zero-mean Gaussian with the given per-dimension variances."""
    var = jnp.asarray(variances, dtype=jnp.float32)
    d = int(var.shape[0])

    def log_prob(x):
        return -0.5 * jnp.sum(x**2 / var) - 0.5 * jnp.sum(jnp.log(2.0 * jnp.pi * var))

    return TargetSpec(log_prob=log_prob, dim=d, name=name)


def gaussian_2d() -> TargetSpec:
    """2D zero-mean Gaussian with diagonal covariance (5, 1)."""
    return diagonal_gaussian(variances=[5.0, 1.0], name="gaussian_2d")


def gaussian_10d() -> TargetSpec:
    """10D zero-mean Gaussian with diagonal variances 1, 2, ..., 10."""
    return diagonal_gaussian(
        variances=list(range(1, 11)),
        name="gaussian_10d",
    )
