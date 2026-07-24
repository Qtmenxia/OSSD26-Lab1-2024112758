import tempfile
import unittest
from pathlib import Path

import numpy as np

from tsp_homework1 import (
    Config,
    generate_cities,
    route_length,
    run_experiment,
    two_opt_delta,
)


class TSPHomeworkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = Config(iterations=2_000)
        self.cities = generate_cities(self.config)

    def test_generated_city_shape_and_range(self) -> None:
        self.assertEqual(self.cities.shape, (100, 2))
        self.assertTrue(np.all(self.cities >= 0))
        self.assertTrue(np.all(self.cities < 10))

    def test_route_length_is_closed_and_positive(self) -> None:
        route = np.arange(self.config.n_cities)
        length = route_length(route, self.cities, self.config.metric)
        self.assertGreater(length, 0)

    def test_two_opt_delta_matches_full_recalculation(self) -> None:
        route = np.arange(self.config.n_cities)
        i, j = 7, 61
        old_length = route_length(route, self.cities, self.config.metric)
        delta = two_opt_delta(route, self.cities, i, j, self.config.metric)

        candidate = route.copy()
        candidate[i : j + 1] = candidate[i : j + 1][::-1]
        new_length = route_length(candidate, self.cities, self.config.metric)

        self.assertAlmostEqual(old_length + delta, new_length, places=9)

    def test_default_full_experiment_improves_baseline(self) -> None:
        config = Config()
        with tempfile.TemporaryDirectory() as directory:
            summary = run_experiment(config, Path(directory))
        self.assertLess(
            summary["improved"]["route_length"],
            summary["baseline"]["route_length"],
        )


if __name__ == "__main__":
    unittest.main()
