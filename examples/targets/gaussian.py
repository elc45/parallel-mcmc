import jax
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

def gaussian_100d() -> TargetSpec:
    """100D zero-mean Gaussian with diagonal variances 1, 2, ..., 100."""
    return diagonal_gaussian(
        variances=list(range(1, 101)),
        name="gaussian_100d",
    )


def toeplitz_gaussian(*, ndims: int, rho: float = 0.9, name: str | None = None) -> TargetSpec:
    """Zero-mean Gaussian with exponential Toeplitz covariance ``Sigma_ij = rho^|i-j|``."""
    if not 0.0 < rho < 1.0:
        raise ValueError(f"toeplitz rho must lie in (0, 1), got {rho}")
    indices = jnp.arange(ndims, dtype=jnp.float64)
    cov = rho ** jnp.abs(indices[:, None] - indices[None, :])
    chol = jnp.linalg.cholesky(cov)
    log_det = 2.0 * jnp.sum(jnp.log(jnp.diag(chol)))
    norm_const = -0.5 * (ndims * jnp.log(2.0 * jnp.pi) + log_det)

    def log_prob(x):
        whitened = jax.scipy.linalg.solve_triangular(chol, x, lower=True)
        return norm_const - 0.5 * jnp.dot(whitened, whitened)

    return TargetSpec(
        log_prob=log_prob,
        dim=ndims,
        name=name or f"toeplitz_gaussian_{ndims}d",
    )


def toeplitz_gaussian_10d() -> TargetSpec:
    """10D zero-mean Gaussian with Toeplitz covariance (rho=0.9)."""
    return toeplitz_gaussian(ndims=10, rho=0.9, name="toeplitz_gaussian_10d")


def toeplitz_gaussian_100d() -> TargetSpec:
    """100D zero-mean Gaussian with Toeplitz covariance (rho=0.9)."""
    return toeplitz_gaussian(ndims=1000, rho=0.9, name="toeplitz_gaussian_100d")

def toeplitz_gaussian_10000d() -> TargetSpec:
    """10000D zero-mean Gaussian with Toeplitz covariance (rho=0.9)."""
    return toeplitz_gaussian(ndims=10000, rho=0.9, name="toeplitz_gaussian_10000d")