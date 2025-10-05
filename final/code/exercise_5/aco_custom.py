import os
import json
from typing import List, Tuple

try:
    import ioh
except Exception as e:
    ioh = None

import numpy as np

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _evaluate(problem, x: np.ndarray) -> float:
    return float(problem(x))

def first_improvement_local_search(problem, x: np.ndarray, rng: np.random.Generator, max_checks: int = 4):
    """Budget-aware 1-bit first-improvement local search (capped checks)."""
    n = len(x)
    order = rng.permutation(n)
    base_f = _evaluate(problem, x)
    checks = 0
    for i in order:
        if checks >= max_checks:
            break
        y = x.copy()
        y[i] ^= 1
        fy = _evaluate(problem, y)
        checks += 1
        if fy > base_f:
            return y, fy, checks
    return x, base_f, checks

def run_aco(
    problem_id: int,
    n: int,
    budget: int,
    seed: int,
    outdir: str,
    ants: int = 10,
    rho: float = 0.1,
    tau_min: float = 0.08,
    tau_max: float = 0.92,
    alpha0: float = 0.5,
    alpha_min: float = 0.2,
    elite_frac: float = 0.3,
    best_so_far_rate: float = 0.15,
    use_best_so_far: bool = True,
    use_local_search: bool = True,
    ls_max_checks: int = 4,      # smaller so 200-eval budget yields more iterations
    stall_window: int = 8,
    restart_alpha: float = 0.35,
) -> Tuple[np.ndarray, float]:
    if ioh is None:
        raise ImportError("ioh is not available. Please `pip install ioh`.")

    # Set up problem
    problem = ioh.get_problem(problem_id, dimension=n, instance=1, problem_class=ioh.ProblemClass.PBO)
    rng = np.random.default_rng(seed)
    _ensure_dir(outdir)

    # Pheromones start neutral
    tau = np.full(n, 0.5, dtype=float)

    # Trackers
    best_x = (rng.random(n) < tau).astype(int)
    best_f = _evaluate(problem, best_x)
    history: List[Tuple[int, float]] = [(1, best_f)]
    evals = 1

    # Estimate iteration count to schedule alpha
    approx_iter_budget = max(1, budget // (ants + ls_max_checks))
    it = 0
    since_improv = 0

    while evals < budget:
        it += 1

        # Construct & evaluate ants
        X = (rng.random((ants, n)) < tau).astype(int)
        fitness = np.empty(ants, dtype=float)

        for a in range(ants):
            fx = _evaluate(problem, X[a])
            evals += 1
            fitness[a] = fx
            if fx > best_f:
                best_f, best_x = fx, X[a].copy()
                since_improv = 0
                # LOG as soon as we improve (more points -> non-flat curve)
                history.append((evals, float(best_f)))
            if evals >= budget:
                break

        # Local search on iteration-best ONLY (first-improvement, capped)
        if use_local_search and evals < budget:
            ib_idx = int(np.argmax(fitness))
            ib_x = X[ib_idx].copy()
            y, fy, used = first_improvement_local_search(problem, ib_x, rng, max_checks=ls_max_checks)
            evals += used
            if fy > fitness[ib_idx]:
                fitness[ib_idx] = fy
                X[ib_idx] = y
                if fy > best_f:
                    best_f, best_x = fy, y.copy()
                    since_improv = 0
                    history.append((evals, float(best_f)))  # LOG improvement from LS too

        # Evaporate
        tau = (1.0 - rho) * tau

        # Rank-based deposit on top-q ants
        q = max(1, int(np.ceil(elite_frac * ants)))
        order = np.argsort(fitness)
        idx = order[-q:]
        # Adaptive alpha schedule
        t = min(1.0, it / max(1, approx_iter_budget))
        alpha = alpha0 + (alpha_min - alpha0) * t

        ranks = np.arange(1, q + 1, dtype=float)
        w = ranks / ranks.sum()

        def deposit(tau_vec, x_bits, rate):
            return np.clip(tau_vec + rate * (x_bits - tau_vec), tau_min, tau_max)

        for j, wj in zip(idx, w):
            tau = deposit(tau, X[j], rate=alpha * wj)
        if use_best_so_far:
            tau = deposit(tau, best_x, rate=best_so_far_rate)

        # Record the end-of-iteration point too (keeps final value on grid)
        history.append((evals, float(best_f)))
        since_improv += 1

        # Soft restart if stuck
        if since_improv >= stall_window:
            tau = (1 - restart_alpha) * tau + restart_alpha * 0.5
            since_improv = 0

    # Save JSON (stable filename)
    result = {
        "algorithm": "ACO-Custom-Optimised",
        "problem_id": problem_id,
        "dimension": n,
        "budget": budget,
        "seed": seed,
        "ants": ants,
        "rho": rho,
        "tau_min": tau_min,
        "tau_max": tau_max,
        "alpha0": alpha0,
        "alpha_min": alpha_min,
        "elite_frac": elite_frac,
        "best_so_far_rate": best_so_far_rate,
        "use_best_so_far": use_best_so_far,
        "use_local_search": use_local_search,
        "ls_max_checks": ls_max_checks,
        "stall_window": stall_window,
        "restart_alpha": restart_alpha,
        "best_solution": best_x.tolist(),
        "best_fitness": float(best_f),
        "history": history,
    }

    fname = os.path.join(outdir, f"aco_p{problem_id}_d{n}_s{seed}.json")
    with open(fname, "w") as f:
        json.dump(result, f, indent=2)

    return best_x, best_f
