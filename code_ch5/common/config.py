"""Configuration values matching Chapter 5 of DAQRNN.tex.

Changes from the previous version (relevant to PIW):
- ``da_iterations`` is set to 200 for every censoring rate for the current
  coverage sensitivity run. This gives the percentile-based PI many more
  bootstrap/DA samples than the original 20/50 schedule.
- ``distributed_da_iterations`` keeps the original distributed-section
  schedule, 20 for 25% censoring and 50 for 50% censoring, because Section 5.5
  reports QL/REE and timing rather than PI calibration.
- ``bootstrap_reps = 0`` means the uncensored benchmark interval uses the same
  number of bootstrap/iteration predictions as the DA method's ``S``. Passing a
  larger value is only a denominator-stability sensitivity check.

Override ``--S`` in the section scripts if you want to temporarily run a
smaller or larger sensitivity setting.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable

import numpy as np


TARGET_TAUS: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 0.90)
J_GRID: tuple[int, ...] = tuple(range(1, 11))
LAMBDA_GRID: tuple[float, ...] = tuple(np.round(np.arange(0.01, 0.101, 0.01), 2))

CENTRAL_N_TRAIN = 500
CENTRAL_N_TEST = 500
CENTRAL_N_REP = 10

DIST_N_TRAIN = 200_000
DIST_N_TEST = 1_000
DIST_N_REP = 10
DIST_WORKERS: tuple[int, ...] = (10, 20, 50)
WORKER_SENSITIVITY_K: tuple[int, ...] = (5, 10, 20, 50, 100)
ILEA_ROUNDS_GRID: tuple[int, ...] = (1, 2, 3, 5, 8, 10)

SCENARIOS: tuple[int, ...] = (1, 2)
ERRORS: tuple[str, ...] = ("normal", "t3")
CENSOR_TYPES: tuple[str, ...] = ("left", "right", "interval")
CENSOR_RATES: tuple[float, ...] = (0.25, 0.50)


@dataclass(frozen=True)
class LevelDist:
    """Distribution used for one censoring boundary."""

    name: str
    a: float
    b: float | None = None


@dataclass(frozen=True)
class CensoringSpec:
    """Censoring-level distributions for one table row in Chapter 5."""

    left: LevelDist | None = None
    right: LevelDist | None = None


def _normal(mean: float, sd: float) -> LevelDist:
    return LevelDist("normal", mean, sd)


def _exp(rate: float) -> LevelDist:
    return LevelDist("exp", rate)


CENSORING_SPECS: dict[tuple[int, str, str, float], CensoringSpec] = {
    # Scenario 1, normal errors
    (1, "normal", "left", 0.25): CensoringSpec(left=_normal(-0.9, 2.0)),
    (1, "normal", "left", 0.50): CensoringSpec(left=_normal(0.6, 2.0)),
    (1, "normal", "right", 0.25): CensoringSpec(right=_normal(2.2, 2.0)),
    (1, "normal", "right", 0.50): CensoringSpec(right=_normal(0.75, 2.0)),
    (1, "normal", "interval", 0.25): CensoringSpec(left=_normal(-0.5, 2.0), right=_normal(0.0, 2.0)),
    (1, "normal", "interval", 0.50): CensoringSpec(left=_normal(-1.5, 2.0), right=_normal(1.5, 2.0)),
    # Scenario 1, t(3) errors
    (1, "t3", "left", 0.25): CensoringSpec(left=_normal(-1.0, 2.0)),
    (1, "t3", "left", 0.50): CensoringSpec(left=_normal(0.5, 2.0)),
    (1, "t3", "right", 0.25): CensoringSpec(right=_normal(2.2, 2.0)),
    (1, "t3", "right", 0.50): CensoringSpec(right=_normal(0.75, 2.0)),
    (1, "t3", "interval", 0.25): CensoringSpec(left=_normal(-0.5, 2.0), right=_normal(0.0, 2.0)),
    (1, "t3", "interval", 0.50): CensoringSpec(left=_normal(-1.5, 2.0), right=_normal(1.5, 2.0)),
    # Scenario 2, normal errors
    (2, "normal", "left", 0.25): CensoringSpec(left=_exp(1.45)),
    (2, "normal", "left", 0.50): CensoringSpec(left=_exp(0.60)),
    (2, "normal", "right", 0.25): CensoringSpec(right=_exp(0.25)),
    (2, "normal", "right", 0.50): CensoringSpec(right=_exp(0.65)),
    (2, "normal", "interval", 0.25): CensoringSpec(left=_exp(1.35), right=_exp(0.85)),
    (2, "normal", "interval", 0.50): CensoringSpec(left=_exp(1.85), right=_exp(0.35)),
    # Scenario 2, t(3) errors
    (2, "t3", "left", 0.25): CensoringSpec(left=_exp(1.45)),
    (2, "t3", "left", 0.50): CensoringSpec(left=_exp(0.65)),
    (2, "t3", "right", 0.25): CensoringSpec(right=_exp(0.25)),
    (2, "t3", "right", 0.50): CensoringSpec(right=_exp(0.60)),
    (2, "t3", "interval", 0.25): CensoringSpec(left=_exp(1.35), right=_exp(0.85)),
    (2, "t3", "interval", 0.50): CensoringSpec(left=_exp(1.45), right=_exp(0.30)),
}


def central_settings() -> Iterable[tuple[int, str, str, float]]:
    return product(SCENARIOS, ERRORS, CENSOR_TYPES, CENSOR_RATES)


def right_censoring_settings() -> Iterable[tuple[int, str, str, float]]:
    return product(SCENARIOS, ERRORS, ("right",), CENSOR_RATES)


def da_iterations(censor_rate: float) -> int:
    # Coverage sensitivity default: use the same large PI sample count for
    # 25% and 50% censoring so percentile intervals are less Monte Carlo noisy.
    return 200


def distributed_da_iterations(censor_rate: float) -> int:
    # Original Section 5.5 schedule. Distributed tables focus on QL_ratio, REE,
    # and time/RACT, so the expensive S=200 PI calibration setting is not used.
    return 20 if censor_rate <= 0.25 else 50


def quick_training_options() -> dict[str, int]:
    return {"max_iter": 35, "ebic_max_iter": 25, "bootstrap_reps": 0}


def paper_training_options() -> dict[str, int]:
    return {"max_iter": 250, "ebic_max_iter": 120, "bootstrap_reps": 0}
