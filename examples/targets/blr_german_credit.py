from pathlib import Path

import jax.numpy as jnp
import numpy as np

from .spec import TargetSpec

_REPO_ROOT = Path(__file__).resolve().parents[2]


def blr_german_credit(
    *,
    data_dir: str | Path = "data",
    sigma_blr: float = 1.0,
) -> TargetSpec:
    data_path = Path(data_dir)
    if not data_path.is_absolute():
        data_path = _REPO_ROOT / data_path

    X = jnp.asarray(np.loadtxt(data_path / "X.txt"))
    y = jnp.asarray(np.loadtxt(data_path / "y.txt"))
    d = int(X.shape[1])

    def log_prob(beta):
        logits = X @ beta
        lp = (
            -0.5 * jnp.sum((beta / sigma_blr) ** 2)
            - d * jnp.log(sigma_blr)
            - 0.5 * d * jnp.log(2.0 * jnp.pi)
        )
        lp += jnp.sum(y * logits - jnp.logaddexp(0.0, logits))
        return lp

    return TargetSpec(log_prob=log_prob, dim=d, name="blr_german_credit")
