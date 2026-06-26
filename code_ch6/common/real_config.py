"""Configuration for Chapter 6 real-data experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .config import LevelDist


TARGET_TAUS: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 0.90)
CENSOR_TYPES: tuple[str, ...] = ("left", "right", "interval")
CENSOR_RATES: tuple[float, ...] = (0.25, 0.50)

CENTRAL_DATASETS: tuple[str, ...] = ("boston", "gilgais")
CENTRAL_CENSOR_RATE = 0.25
CENTRAL_N_REP = 200
CENTRAL_TRAIN_FRACTION = 0.80
CENTRAL_DA_ITERATIONS = 200

HOUSEHOLD_N_TRAIN = 1_000_000
HOUSEHOLD_N_REP = 1
HOUSEHOLD_WORKERS: tuple[int, ...] = (10, 20, 50)
HOUSEHOLD_ILEA_ROUNDS = 1
HOUSEHOLD_TAU_GRID_SIZE = 100


@dataclass(frozen=True)
class RealCensoringSpec:
    """Censoring-level distributions for real-data experiments.

    ``left`` and ``right`` are interpreted on the standardized response scale.
    For interval censoring both boundaries are sampled and then ordered.
    """

    left: LevelDist | None = None
    right: LevelDist | None = None


def normal_level(mean: float, sd: float) -> LevelDist:
    return LevelDist("normal", float(mean), float(sd))


CENTRAL_CENSORING_SPECS: dict[tuple[str, str], RealCensoringSpec] = {
    ("boston", "left"): RealCensoringSpec(left=normal_level(-1.0, 1.0)),
    ("boston", "right"): RealCensoringSpec(right=normal_level(1.0, 1.0)),
    ("boston", "interval"): RealCensoringSpec(left=normal_level(0.2, 1.0), right=normal_level(1.0, 1.0)),
    ("gilgais", "left"): RealCensoringSpec(left=normal_level(-1.2, 1.0)),
    ("gilgais", "right"): RealCensoringSpec(right=normal_level(1.0, 1.0)),
    ("gilgais", "interval"): RealCensoringSpec(left=normal_level(-0.5, 1.0), right=normal_level(0.0, 1.0)),
}


HOUSEHOLD_CENSORING_SPECS: dict[tuple[str, float], RealCensoringSpec] = {
    # Boundaries are generated on the standardized Global_active_power scale.
    # The response is strongly right-skewed, so the means are intentionally
    # asymmetric. The nominal 25% and 50% labels are target settings; the
    # realised censoring rate is saved in censoring_summary.csv for reporting.
    ("left", 0.25): RealCensoringSpec(left=normal_level(-0.75, 0.35)),
    ("left", 0.50): RealCensoringSpec(left=normal_level(-0.25, 0.35)),
    ("right", 0.25): RealCensoringSpec(right=normal_level(0.55, 0.50)),
    ("right", 0.50): RealCensoringSpec(right=normal_level(-0.25, 0.35)),
    ("interval", 0.25): RealCensoringSpec(left=normal_level(-0.55, 0.35), right=normal_level(0.00, 0.35)),
    ("interval", 0.50): RealCensoringSpec(left=normal_level(-0.75, 0.35), right=normal_level(0.55, 0.50)),
}


def distributed_da_iterations(censor_rate: float) -> int:
    return 20 if float(censor_rate) <= 0.25 else 50


def fixed_tau_grid(size: int = HOUSEHOLD_TAU_GRID_SIZE) -> np.ndarray:
    """Fixed DA imputation grid for large real data.

    The simulation code uses roughly sqrt(n) grid points. With one million
    observations this would create very large candidate-prediction matrices.
    For the household data we keep a fixed dense grid, which preserves the
    random-quantile imputation idea while keeping memory predictable.
    """

    k = max(int(size), len(TARGET_TAUS))
    grid = np.arange(1, k + 1, dtype=float) / (k + 1.0)
    return np.unique(np.concatenate([grid, np.asarray(TARGET_TAUS, dtype=float)]))


def central_settings(datasets: Iterable[str] = CENTRAL_DATASETS):
    for dataset in datasets:
        for censor_type in CENSOR_TYPES:
            yield dataset, censor_type, CENTRAL_CENSOR_RATE


def distributed_settings():
    for censor_rate in CENSOR_RATES:
        for censor_type in CENSOR_TYPES:
            yield censor_type, censor_rate
