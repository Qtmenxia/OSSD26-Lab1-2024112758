"""
Home Task 1 - TSP improvement with 2-opt simulated annealing.

This file provides:
1. A baseline simulated-annealing solver using a simple two-city swap.
2. An improved solver using a 2-opt segment reversal.
3. Reproducible comparison plots and a JSON summary.

The default experiment keeps the lecture settings:
- 100 cities
- coordinates generated in [0, 10)
- initial temperature = 100
- end temperature = 0
- 100,000 iterations
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MetricName = Literal["euclidean", "lecture"]


@dataclass(frozen=True)
class Config:
    n_cities: int = 100
    initial_temperature: float = 100.0
    end_temperature: float = 0.0
    iterations: int = 100_000
    city_seed: int = 42
    optimizer_seed: int = 43
    coordinate_limit: float = 10.0
    metric: MetricName = "euclidean"


@dataclass
class AlgorithmResult:
    name: str
    route: np.ndarray
    length: float
    accepted_moves: int
    improving_moves: int
    history_iterations: list[int]
    history_best_lengths: list[float]


def generate_cities(config: Config) -> np.ndarray:
    """Use the same point-generation rule as the lecture."""
    np.random.seed(config.city_seed)
    return np.random.rand(config.n_cities, 2) * config.coordinate_limit


def point_distance(a: np.ndarray, b: np.ndarray, metric: MetricName) -> float:
    if metric == "euclidean":
        return float(np.linalg.norm(a - b))
    if metric == "lecture":
        # Exact behavior of the distance function shown in the slides.
        return float(np.sqrt((a - b) ** 2).sum())
    raise ValueError(f"Unsupported metric: {metric}")


def route_length(
    route: np.ndarray,
    cities: np.ndarray,
    metric: MetricName = "euclidean",
) -> float:
    if len(route) != len(cities):
        raise ValueError("Route must contain exactly one index for each city.")
    if len(np.unique(route)) != len(route):
        raise ValueError("Route must be a permutation without duplicate cities.")

    points = cities[route]
    next_points = np.roll(points, -1, axis=0)

    if metric == "euclidean":
        return float(np.linalg.norm(points - next_points, axis=1).sum())
    if metric == "lecture":
        return float(np.abs(points - next_points).sum())
    raise ValueError(f"Unsupported metric: {metric}")


def temperature_at(initial_temperature: float, iteration: int) -> float:
    """Cooling schedule from the lecture."""
    return initial_temperature * 0.1 / iteration


def accept_move(delta: float, temperature: float, rng: np.random.Generator) -> bool:
    if delta <= 0:
        return True
    if temperature <= 0:
        return False
    probability = math.exp(-delta / temperature)
    return bool(rng.random() <= probability)


def simple_swap_candidate(
    route: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    candidate = route.copy()
    i, j = rng.choice(len(route), size=2, replace=False)
    candidate[i], candidate[j] = candidate[j], candidate[i]
    return candidate


def two_opt_delta(
    route: np.ndarray,
    cities: np.ndarray,
    i: int,
    j: int,
    metric: MetricName,
) -> float:
    """
    Return the length change caused by reversing route[i:j+1].

    Only two boundary edges change in a symmetric TSP, so the delta
    can be computed without recalculating the complete route.
    """
    n = len(route)

    # Reversing the entire closed cycle only changes orientation.
    if i == 0 and j == n - 1:
        return 0.0

    a = route[(i - 1) % n]
    b = route[i]
    c = route[j]
    d = route[(j + 1) % n]

    old_edges = (
        point_distance(cities[a], cities[b], metric)
        + point_distance(cities[c], cities[d], metric)
    )
    new_edges = (
        point_distance(cities[a], cities[c], metric)
        + point_distance(cities[b], cities[d], metric)
    )
    return new_edges - old_edges


def run_simple_swap_sa(cities: np.ndarray, config: Config) -> AlgorithmResult:
    rng = np.random.default_rng(config.optimizer_seed)
    route = np.arange(config.n_cities)
    current_length = route_length(route, cities, config.metric)

    best_route = route.copy()
    best_length = current_length
    accepted_moves = 0
    improving_moves = 0

    history_iterations = [0]
    history_best_lengths = [best_length]
    record_every = max(1, config.iterations // 200)

    for k in range(1, config.iterations + 1):
        candidate = simple_swap_candidate(route, rng)
        candidate_length = route_length(candidate, cities, config.metric)
        delta = candidate_length - current_length
        temperature = temperature_at(config.initial_temperature, k)

        if accept_move(delta, temperature, rng):
            route = candidate
            current_length = candidate_length
            accepted_moves += 1

            if current_length < best_length:
                best_route = route.copy()
                best_length = current_length
                improving_moves += 1

        if k % record_every == 0 or k == config.iterations:
            history_iterations.append(k)
            history_best_lengths.append(best_length)

        if temperature <= config.end_temperature:
            break

    return AlgorithmResult(
        name="Simple-swap simulated annealing",
        route=best_route,
        length=best_length,
        accepted_moves=accepted_moves,
        improving_moves=improving_moves,
        history_iterations=history_iterations,
        history_best_lengths=history_best_lengths,
    )


def run_two_opt_sa(cities: np.ndarray, config: Config) -> AlgorithmResult:
    rng = np.random.default_rng(config.optimizer_seed)
    route = np.arange(config.n_cities)
    current_length = route_length(route, cities, config.metric)

    best_route = route.copy()
    best_length = current_length
    accepted_moves = 0
    improving_moves = 0

    history_iterations = [0]
    history_best_lengths = [best_length]
    record_every = max(1, config.iterations // 200)

    for k in range(1, config.iterations + 1):
        i, j = sorted(rng.choice(config.n_cities, size=2, replace=False))

        if i == 0 and j == config.n_cities - 1:
            continue

        delta = two_opt_delta(route, cities, i, j, config.metric)
        temperature = temperature_at(config.initial_temperature, k)

        if accept_move(delta, temperature, rng):
            route[i : j + 1] = route[i : j + 1][::-1]
            current_length += delta
            accepted_moves += 1

            if current_length < best_length:
                best_route = route.copy()
                best_length = current_length
                improving_moves += 1

        if k % record_every == 0 or k == config.iterations:
            history_iterations.append(k)
            history_best_lengths.append(best_length)

        if temperature <= config.end_temperature:
            break

    # Recalculate once to remove accumulated floating-point delta error.
    best_length = route_length(best_route, cities, config.metric)

    return AlgorithmResult(
        name="2-opt simulated annealing",
        route=best_route,
        length=best_length,
        accepted_moves=accepted_moves,
        improving_moves=improving_moves,
        history_iterations=history_iterations,
        history_best_lengths=history_best_lengths,
    )


def save_route_plot(
    cities: np.ndarray,
    route: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    closed_route = np.append(route, route[0])
    points = cities[closed_route]

    plt.figure(figsize=(9, 7))
    plt.plot(points[:, 0], points[:, 1], "--o", markersize=3, linewidth=0.9)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=170)
    plt.close()


def save_length_comparison(
    labels: list[str],
    lengths: list[float],
    output_path: Path,
) -> None:
    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, lengths)

    for bar, value in zip(bars, lengths):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}",
            ha="center",
            va="bottom",
        )

    plt.ylabel("Closed route length")
    plt.title("TSP route-length comparison")
    plt.xticks(rotation=12, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=170)
    plt.close()


def save_convergence_plot(
    baseline: AlgorithmResult,
    improved: AlgorithmResult,
    output_path: Path,
) -> None:
    plt.figure(figsize=(9, 5.5))
    plt.plot(
        baseline.history_iterations,
        baseline.history_best_lengths,
        label="Simple swap",
    )
    plt.plot(
        improved.history_iterations,
        improved.history_best_lengths,
        label="2-opt",
    )
    plt.xlabel("Iteration")
    plt.ylabel("Best route length found")
    plt.title("Convergence comparison")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=170)
    plt.close()


def run_experiment(config: Config, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    cities = generate_cities(config)
    initial_route = np.arange(config.n_cities)
    initial_length = route_length(initial_route, cities, config.metric)

    baseline = run_simple_swap_sa(cities, config)
    improved = run_two_opt_sa(cities, config)

    if improved.length >= baseline.length:
        raise RuntimeError(
            "For the selected seed, the 2-opt result was not better than the "
            "simple-swap baseline. Try another optimizer seed."
        )

    save_route_plot(
        cities,
        initial_route,
        f"Initial route (length = {initial_length:.3f})",
        output_dir / "initial_route.png",
    )
    save_route_plot(
        cities,
        baseline.route,
        f"Simple-swap SA (length = {baseline.length:.3f})",
        output_dir / "baseline_swap_route.png",
    )
    save_route_plot(
        cities,
        improved.route,
        f"Improved 2-opt SA (length = {improved.length:.3f})",
        output_dir / "improved_2opt_route.png",
    )
    save_length_comparison(
        ["Initial", "Simple swap", "2-opt"],
        [initial_length, baseline.length, improved.length],
        output_dir / "route_length_comparison.png",
    )
    save_convergence_plot(
        baseline,
        improved,
        output_dir / "convergence_comparison.png",
    )

    summary = {
        "config": asdict(config),
        "initial_route_length": initial_length,
        "baseline": {
            "algorithm": baseline.name,
            "route_length": baseline.length,
            "accepted_moves": baseline.accepted_moves,
            "improving_moves": baseline.improving_moves,
            "route": baseline.route.tolist(),
        },
        "improved": {
            "algorithm": improved.name,
            "route_length": improved.length,
            "accepted_moves": improved.accepted_moves,
            "improving_moves": improved.improving_moves,
            "route": improved.route.tolist(),
        },
        "improvement": {
            "absolute_vs_baseline": baseline.length - improved.length,
            "percent_vs_baseline": (
                (baseline.length - improved.length) / baseline.length * 100
            ),
            "percent_vs_initial": (
                (initial_length - improved.length) / initial_length * 100
            ),
        },
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare simple-swap and 2-opt simulated annealing for TSP."
    )
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--city-seed", type=int, default=42)
    parser.add_argument("--optimizer-seed", type=int, default=43)
    parser.add_argument("--iterations", type=int, default=100_000)
    parser.add_argument(
        "--metric",
        choices=["euclidean", "lecture"],
        default="euclidean",
        help=(
            "euclidean: standard straight-line TSP distance; "
            "lecture: exact metric behavior shown in the slides"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config(
        city_seed=args.city_seed,
        optimizer_seed=args.optimizer_seed,
        iterations=args.iterations,
        metric=args.metric,
    )

    summary = run_experiment(config, Path(args.output_dir))

    print("Home Task 1 completed.")
    print(f"Initial route: {summary['initial_route_length']:.6f}")
    print(f"Simple-swap SA: {summary['baseline']['route_length']:.6f}")
    print(f"2-opt SA: {summary['improved']['route_length']:.6f}")
    print(
        "2-opt improvement over baseline: "
        f"{summary['improvement']['percent_vs_baseline']:.2f}%"
    )
    print(f"Files saved in: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
