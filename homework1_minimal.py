import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def metric(a: np.ndarray, b: np.ndarray) -> float:
    """Distance function used in the lecture slides."""
    distance = (a - b) ** 2
    distance = np.sqrt(distance)
    distance = np.sum(distance)
    return float(distance)


def calculate_energy(sequence: np.ndarray, cities: np.ndarray) -> float:
    """Calculate the closed route length."""
    energy = 0.0
    for i in range(len(sequence) - 1):
        energy += metric(cities[sequence[i]], cities[sequence[i + 1]])

    energy += metric(cities[sequence[-1]], cities[sequence[0]])
    return energy


def decrease_temperature(initial_temperature: float, k: int) -> float:
    return initial_temperature * 0.1 / k


def get_transition_probability(delta_energy: float, temperature: float) -> float:
    if temperature <= 0:
        return 0.0
    return float(np.exp(-delta_energy / temperature))


def is_transition(probability: float) -> bool:
    return bool(np.random.rand() <= probability)


def generate_state_candidate(sequence: np.ndarray) -> np.ndarray:
    """
    Generate a candidate using 2-opt.

    Example:
        [1, 2, 3, 4, 5, 6, 7]
        i = 1, j = 4
        -> [1, 5, 4, 3, 2, 6, 7]
    """
    candidate = sequence.copy()
    n = len(candidate)

    i, j = np.sort(np.random.choice(n, size=2, replace=False))
    candidate[i : j + 1] = candidate[i : j + 1][::-1]
    return candidate


def optimise_route(
    cities: np.ndarray,
    initial_temperature: float,
    end_temperature: float,
    max_iterations: int = 100_000,
) -> tuple[np.ndarray, float]:
    n_cities = cities.shape[0]
    state = np.arange(n_cities)
    current_energy = calculate_energy(state, cities)

    # Keep the best route ever seen instead of only the final accepted route.
    best_state = state.copy()
    best_energy = current_energy

    print(f"Initial route length: {current_energy:.6f}")

    temperature = initial_temperature

    for k in range(1, max_iterations + 1):
        state_candidate = generate_state_candidate(state)
        candidate_energy = calculate_energy(state_candidate, cities)
        delta_energy = candidate_energy - current_energy

        if delta_energy < 0:
            accept = True
        else:
            probability = get_transition_probability(delta_energy, temperature)
            accept = is_transition(probability)

        if accept:
            state = state_candidate
            current_energy = candidate_energy

            if current_energy < best_energy:
                best_state = state.copy()
                best_energy = current_energy

        temperature = decrease_temperature(initial_temperature, k)
        if temperature <= end_temperature:
            break

    print(f"Final best route length: {best_energy:.6f}")
    return best_state, best_energy


def save_route_plot(
    cities: np.ndarray,
    route: np.ndarray,
    title: str,
    output_path: str,
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
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    # Same settings as the lecture.
    n_cities = 100
    initial_temperature = 100
    end_temperature = 0

    # A fixed seed makes the homework result reproducible.
    np.random.seed(42)

    # Same point-generation rule as the lecture.
    cities = np.random.rand(n_cities, 2) * 10

    initial_route = np.arange(n_cities)
    save_route_plot(
        cities,
        initial_route,
        "Initial route",
        "minimal_initial_route.png",
    )

    best_route, best_length = optimise_route(
        cities,
        initial_temperature,
        end_temperature,
    )

    save_route_plot(
        cities,
        best_route,
        f"Optimized route with 2-opt (length = {best_length:.3f})",
        "minimal_2opt_route.png",
    )


if __name__ == "__main__":
    main()
