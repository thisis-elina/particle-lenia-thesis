import numpy as np

def stability_score(points):
    """Measure particle movement stability (low movement = high stability)."""
    diffs = np.diff(points, axis=0)
    return 1.0 / (1.0 + np.mean(np.linalg.norm(diffs, axis=1)))

def diversity_score(points):
    """Measure spatial spread / diversity of particles."""
    cov = np.cov(points.T)
    return np.linalg.det(cov)

def goal_completion(points, goal_pos):
    """Compute fraction of particles close to a goal."""
    distances = np.linalg.norm(points - goal_pos, axis=1)
    return np.mean(distances < 1.0)
