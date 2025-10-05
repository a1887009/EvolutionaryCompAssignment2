import os
import json
from datetime import datetime
from typing import List, Tuple

try:
    import ioh
except Exception as e:
    ioh = None

import numpy as np

# ------------------------------
# Custom ACO for PBO (bitstrings)
# ------------------------------
# - Pheromone vector tau in [tau_min, tau_max] per bit
# - Construct solutions by sampling each bit: x_i ~ Bernoulli(tau_i)
# - Evaporation: tau = (1 - rho) * tau
# - Deposit on best ant of the iteration (iteration-best) and optionally best-so-far
# - Optional simple local search: one-step 1-bit improvement (greedy flip)

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _evaluate(problem, x: np.ndarray) -> float:
    # IOH problems are by default minimization for REAL suite, but for PBO
    # they are typically maximization. We stick to problem meta.
    return float(problem(x))

def greedy_1bit_local_search(problem, x: np.ndarray) -> Tuple[np.ndarray, float]:
    """Try all 1-bit flips; return the best improving neighbor if any."""
    best_x = x.copy()
    best_f = _evaluate(problem, x)
    n = len(x)
    for i in range(n):
        y = x.copy()
        y[i] ^= 1  # flip bit
        fy = _evaluate(problem, y)
        if fy > best_f:
            best_f, best_x = fy, y
    return best_x, best_f

def run_aco(
    problem_id: int,
    n: int,
    budget: int,
    seed: int,
    outdir: str,
    ants: int = 20,
    rho: float = 1.0 / 50,   # evaporation rate
    tau_min: float = 0.01,
    tau_max: float = 0.99,
    use_best_so_far: bool = True,
    use_local_search: bool = True,
) -> Tuple[np.ndarray, float]:
    """Run custom ACO on a PBO function using IOHexperimenter problems.

    Logs a compact JSON with (evals, best) history, consistent with GA logs.
    """
    if ioh is None:
        raise ImportError(
            "ioh is not available. Please `pip install ioh` in your environment to run this."
        )

    # Set up IOH problem (PBO)
    problem = ioh.get_problem(problem_id, dimension=n, instance=1, problem_class=ioh.ProblemClass.PBO)

    rng = np.random.default_rng(seed)
    _ensure_dir(outdir)

    # Initialize pheromones to 0.5 (uninformative prior)
    tau = np.ones(n) * 0.5

    # Initialize best trackers
    best_x = (rng.random(n) < tau).astype(int)
    best_f = _evaluate(problem, best_x)

    history: List[Tuple[int, float]] = [(1, best_f)]
    evals = 1

    while evals < budget:
        # Generate ant solutions
        X = (rng.random((ants, n)) < tau).astype(int)
        fitness = np.empty(ants, dtype=float)

        for a in range(ants):
            fx = _evaluate(problem, X[a])
            evals += 1
            # Optional 1-bit local search
            if use_local_search:
                X[a], fx = greedy_1bit_local_search(problem, X[a])
                evals += n  # local search evaluates up to n neighbors in worst case

            fitness[a] = fx
            if fx > best_f:
                best_f, best_x = fx, X[a].copy()
            if evals >= budget:
                break

        # Evaporate
        tau = (1.0 - rho) * tau

        # Reinforce with iteration-best and optionally best-so-far
        ib_idx = int(np.argmax(fitness))
        ib_x = X[ib_idx]

        # Deposit: move tau toward the bit values of the elite solutions
        def deposit(tau_vec, x_bits, amount):
            return np.clip(tau_vec + amount * (x_bits - tau_vec), tau_min, tau_max)

        tau = deposit(tau, ib_x, amount=rho)
        if use_best_so_far:
            tau = deposit(tau, best_x, amount=rho * 0.5)

        history.append((evals, float(best_f)))

    # Save result JSON
    result = {
        "algorithm": "ACO-Custom",
        "problem_id": problem_id,
        "dimension": n,
        "budget": budget,
        "seed": seed,
        "ants": ants,
        "rho": rho,
        "tau_min": tau_min,
        "tau_max": tau_max,
        "use_best_so_far": use_best_so_far,
        "use_local_search": use_local_search,
        "best_solution": best_x.tolist(),
        "best_fitness": float(best_f),
        "history": history,
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = os.path.join(outdir, f"aco_p{problem_id}_d{n}_s{seed}_{timestamp}.json")
    with open(fname, "w") as f:
        json.dump(result, f, indent=2)

    return best_x, best_f
