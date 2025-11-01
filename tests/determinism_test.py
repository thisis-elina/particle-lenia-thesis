import sys, os, numpy as np
# Ensure project root on path when running from tests/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from simulations.particle_lenia_headless import ParticleLeniaSimulation


def main():
    """Determinism test: same config + same seed → identical results.

    Aligns naming with the main project (average_energy, points).
    """
    config = {"mu_k": 4.0, "sigma_k": 1.0, "w_k": 0.022, "c_rep": 1.0,
              "mu_g": 0.6, "sigma_g": 0.15, "dt": 0.1, "point_n": 64}

    sim1 = ParticleLeniaSimulation(config, seed=123)
    average_energy_1 = sim1.run_headless(steps=30)
    points1 = sim1.get_state().copy()

    sim2 = ParticleLeniaSimulation(config, seed=123)
    average_energy_2 = sim2.run_headless(steps=30)
    points2 = sim2.get_state().copy()

    assert np.isclose(average_energy_1, average_energy_2), (average_energy_1, average_energy_2)
    assert np.allclose(points1, points2)
    print("Determinism OK")


if __name__ == "__main__":
    main()
