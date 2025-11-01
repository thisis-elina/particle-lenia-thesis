import sys, os, numpy as np
# Ensure project root on path when running from tests/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.metrics import stability_score, diversity_score, goal_completion


def main():
    """Metric bounds/sanity checks with aligned naming.

    Confirms diversity is ~0 for degenerate points, stability is finite on a
    simple trajectory, and goal completion lies within [0, 1].
    """
    # Degenerate identical points → diversity approximately 0
    points = np.zeros((10, 2), dtype=float)
    diversity = diversity_score(points)
    assert np.isfinite(diversity)
    assert abs(diversity - 0.0) < 1e-9, f"Expected ~0 diversity, got {diversity}"

    # Stability finite for a simple monotone trajectory
    trajectory = np.stack([np.zeros(2), np.ones(2) * 0.1, np.ones(2) * 0.2])
    stability = stability_score(trajectory)
    assert np.isfinite(stability), f"Stability is not finite: {stability}"

    # Goal completion within [0, 1]
    points = np.array([[0, 0], [0.5, 0.5], [2, 2]], dtype=float)
    goal = goal_completion(points, goal_pos=np.array([0, 0], dtype=float))
    assert 0.0 <= goal <= 1.0, f"Goal completion out of bounds: {goal}"

    print("Metrics bounds OK")


if __name__ == "__main__":
    main()
