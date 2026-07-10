"""Plot damp-factor scan results (JSON only — no JAX/GPU deps).

Reads every ``task_*.json`` under ``--results-dir``, groups by ``damp_factor``,
and plots the mean Newton iteration count. When multiple seeds (or replicate
``n_iters`` values) exist for a damp factor, error bars show ±1 sample std.

Run:
    .venv/bin/python examples/plot_nuts_damp_scan.py
    .venv/bin/python examples/plot_nuts_damp_scan.py --results-dir ... --output ...
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCAN_DIR = (
    _REPO_ROOT
    / "experiments"
    / "nuts"
    / "scans"
    / "damp_factor"
    / "blr_german_credit"
)
DEFAULT_STEP_SIZE = 0.05


def _eps_results_dir(step_size: float) -> Path:
    return DEFAULT_SCAN_DIR / "results" / f"eps_{step_size:g}"


def _default_output(results_dir: Path) -> Path:
    return results_dir / "n_iters_vs_damp_factor.png"


def _iters_from_record(record: dict) -> list[float]:
    """Extract all replicate iteration counts from one task JSON."""
    if "n_iters_per_seed" in record:
        return [float(x) for x in record["n_iters_per_seed"]]
    if "n_iters" in record:
        return [float(record["n_iters"])]
    raise KeyError(f"Record missing n_iters / n_iters_per_seed: {record!r}")


def _mean_std(values: list[float]) -> tuple[float, float]:
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, 0.0
    var = sum((x - mean) ** 2 for x in values) / (n - 1)
    return mean, math.sqrt(var)


def plot_damp_scan(
    *,
    results_dir: Path,
    output: Path,
    target: str = "blr_german_credit",
    step_size: float = 0.1,
    chain_length: int = 200,
    expected_n: int | None = None,
) -> None:
    result_files = sorted(results_dir.glob("task_*.json"))
    if not result_files:
        raise FileNotFoundError(f"No task_*.json files found in {results_dir}")

    records = [json.loads(path.read_text()) for path in result_files]

    # Group all replicate n_iters by damp_factor (supports multi-seed JSONs
    # and multiple task files that share a damp factor).
    by_damp: dict[float, list[float]] = defaultdict(list)
    meta_record: dict | None = None
    for record in records:
        damp = float(record["damp_factor"])
        by_damp[damp].extend(_iters_from_record(record))
        if meta_record is None:
            meta_record = record

    damp_factors = sorted(by_damp)
    means: list[float] = []
    stds: list[float] = []
    for damp in damp_factors:
        mean, std = _mean_std(by_damp[damp])
        means.append(mean)
        stds.append(std)

    n_points = len(damp_factors)
    if expected_n is not None and n_points != expected_n:
        print(
            f"Note: found {n_points} damp-factor point(s); "
            f"expected_n={expected_n} (plotting what is present)"
        )

    assert meta_record is not None
    target = str(meta_record.get("target", target))
    step_size = float(meta_record.get("step_size", step_size))
    chain_length = int(meta_record.get("chain_length", chain_length))

    n_seeds = max(len(v) for v in by_damp.values())
    seed_note = f", n_seeds={n_seeds}" if n_seeds > 1 else ""

    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if n_seeds > 1 and any(s > 0 for s in stds):
        ax.errorbar(
            damp_factors,
            means,
            yerr=stds,
            fmt="o-",
            linewidth=1.5,
            markersize=7,
            capsize=3,
            label=f"mean ± std ({n_seeds} seeds)",
        )
        ax.legend()
    else:
        ax.plot(damp_factors, means, "o-", linewidth=1.5, markersize=7)

    ax.set_xlabel("DEER damp factor")
    ax.set_ylabel("Newton iterations to convergence")
    ax.set_title(
        f"Parallel NUTS + DEER ({target}, step_size={step_size:g}, "
        f"chain_length={chain_length}{seed_note})"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {output} ({n_points} damp-factor point(s))")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Directory of task_*.json files (default: .../results/eps_<step-size>).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Plot path (default: <results-dir>/n_iters_vs_damp_factor.png).",
    )
    parser.add_argument("--target", default="blr_german_credit")
    parser.add_argument("--step-size", type=float, default=DEFAULT_STEP_SIZE)
    parser.add_argument("--chain-length", type=int, default=200)
    parser.add_argument(
        "--expected-n",
        type=int,
        default=None,
        help="Optional sanity check: warn if number of damp factors differs.",
    )
    parser.add_argument(
        "--damp-factors",
        type=float,
        nargs="+",
        default=None,
        help="Deprecated alias for --expected-n=len(list); ignored if absent.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    expected_n = args.expected_n
    if expected_n is None and args.damp_factors is not None:
        expected_n = len(args.damp_factors)

    results_dir = (
        args.results_dir.resolve()
        if args.results_dir is not None
        else _eps_results_dir(args.step_size)
    )
    output = (
        args.output.resolve()
        if args.output is not None
        else _default_output(results_dir)
    )

    plot_damp_scan(
        results_dir=results_dir,
        output=output,
        target=args.target,
        step_size=args.step_size,
        chain_length=args.chain_length,
        expected_n=expected_n,
    )


if __name__ == "__main__":
    main()
