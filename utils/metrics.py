"""
Metric utilities used by the headless simulations and experiment tooling.

Definitions (as used in the thesis):
- stability_score(points): higher when particles move less between steps.
- diversity_score(points): determinant of the covariance of point positions (spread).
- goal_completion(points, goal_pos): fraction of particles within radius 1.0 of goal.

Caveats:
- diversity_score can be near-zero if the covariance is singular (collapsed states).
- Ensure inputs are finite; upstream sims should avoid NaNs/Infs.
"""
from __future__ import annotations

import numpy as np


def stability_score(points: np.ndarray) -> float:
    """Compute stability of motion from a trajectory of particle positions.

    Interprets `points` as an ordered sequence of positions (T x 2) and
    returns a scalar in (0, 1], where higher values indicate lower average
    stepwise displacement.

    Parameters
    ----------
    points : np.ndarray
        Array of shape (T, 2) with positions across time (or steps) for a
        representative particle aggregate. In our sims, this is typically
        the stacked positions or the trajectory used for metrics.

    Returns
    -------
    float
        Stability score computed as 1 / (1 + mean( ||Δx|| )).
    """
    diffs = np.diff(points, axis=0)
    mean_step = np.mean(np.linalg.norm(diffs, axis=1)) if len(diffs) > 0 else 0.0
    return 1.0 / (1.0 + mean_step)


def diversity_score(points: np.ndarray) -> float:
    """Measure spatial diversity using the determinant of the covariance matrix.

    Parameters
    ----------
    points : np.ndarray
        Array of shape (N, 2) with particle positions at a given time.

    Returns
    -------
    float
        det(cov(points)). Larger values indicate a broader spread; can be 0 if
        points are collinear or identical.
    """
    if points.size == 0:
        return 0.0
    cov = np.cov(points.T)
    return float(np.linalg.det(cov))


def goal_completion(points: np.ndarray, goal_pos: np.ndarray) -> float:
    """Compute the fraction of particles within radius 1.0 of the goal.

    Parameters
    ----------
    points : np.ndarray
        Array of shape (N, 2) of particle positions.
    goal_pos : np.ndarray
        Array of shape (2,) with goal coordinates.

    Returns
    -------
    float
        Value in [0, 1] representing the proportion of particles "at" the goal.
    """
    if points.size == 0:
        return 0.0
    distances = np.linalg.norm(points - goal_pos, axis=1)
    return float(np.mean(distances < 1.0))
