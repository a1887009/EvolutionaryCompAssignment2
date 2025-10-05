import os
import json
from typing import List, Tuple

try:
    import ioh
except Exception as e:
    ioh = None

import numpy as np

# Utility function to ensure the directory exists

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

# Evaluates the problem at a given solution
def _evaluate(problem, x: np.ndarray) -> float:
    return float(problem(x))

# Computes the average per-bit entropy of the pheromone vector
def _bit_entropy(tau: np.ndarray) -> float:
    """Average per-bit entropy in [0, ln(2)], larger = more uncertain."""
    eps = 1e-12
    t = np.clip(tau, eps, 1 - eps)
    h = -(t * np.log(t) + (1 - t) * np.log(1 - t))
    return float(np.mean(h))

def _first_improvement_ls(problem, x: np.ndarray, rng: np.random.Generator, max_checks: int) -> Tuple[np.ndarray, float, int]:
    """Random-order 1-bit first-improvement; returns (best_x, best_f, evals_used)."""
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

# Main ACO function
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
    alpha0: float = 0.6,
    alpha_min: float = 0.15,
    elite_frac: float = 0.3,
    best_so_far_rate: float = 0.15,
    use_best_so_far: bool = True,
    use_local_search: bool = True,
    ls_max_checks: int = 16,
    stall_window: int = 12,
    restart_alpha: float = 0.35,
    entropy_thresh: float = 0.15,
    log_interval: int = 1000,
) -> Tuple[np.ndarray, float]:
    if ioh is None:
        raise ImportError("ioh is not available. Please `pip install ioh`.")

    # Problem setup and RNG
    problem = ioh.get_problem(problem_id, dimension=n, instance=1, problem_class=ioh.ProblemClass.PBO)
    rng = np.random.default_rng(seed)
    _ensure_dir(outdir)

    # Pheromones start neutral
    tau = np.full(n, 0.5, dtype=float)

    # these are the trackers
    best_x = (rng.random(n) < tau).astype(int)
    best_f = _evaluate(problem, best_x)
    evals = 1
    history: List[Tuple[int, float]] = [(evals, best_f)]

    # Iteration schedule
    # Approximate the number of iterations possible within the budget, accounting for the ants and the LS checks
    approx_iter_budget = max(1, budget // max(1, ants + ls_max_checks))
    it = 0
    since_improv = 0

    # Fixed-interval logging setup
    next_log = min(budget, ((evals // log_interval) + 1) * log_interval)

    while evals < budget:
        it += 1

        # Half samples from Bernoulli, half as their bitwise complements if even.
        k = ants // 2
        base = (rng.random((k, n)) < tau).astype(int)
        comp = 1 - base
        X = np.vstack([base, comp])
        if X.shape[0] < ants:
            # If odd, add one extra Bernoulli sample
            extra = (rng.random((1, n)) < tau).astype(int)
            X = np.vstack([X, extra])

        # Evaluate ants and update best
        fitness = np.full(X.shape[0], -np.inf, dtype=float)
        for a in range(X.shape[0]):
            if evals >= budget:
                break
            fx = _evaluate(problem, X[a])
            evals += 1
            fitness[a] = fx
            if fx > best_f:
                best_f, best_x = fx, X[a].copy()
                since_improv = 0
                history.append((evals, best_f))
            # fixed-interval logging
            while evals >= next_log and next_log <= budget:
                history.append((next_log, best_f))
                next_log += log_interval

        if evals >= budget:
            break
        # local search on the best ant of this iteration
        if use_local_search:
            ib_idx = int(np.argmax(fitness))
            remaining = max(0, budget - evals)
            checks = min(ls_max_checks, remaining)
            if checks > 0:
                y, fy, used = _first_improvement_ls(problem, X[ib_idx], rng, checks)
                evals += used
                if fy > fitness[ib_idx]:
                    fitness[ib_idx] = fy
                    X[ib_idx] = y
                    if fy > best_f:
                        best_f, best_x = fy, y.copy()
                        since_improv = 0
                        history.append((evals, best_f))
                while evals >= next_log and next_log <= budget:
                    history.append((next_log, best_f))
                    next_log += log_interval

        if evals >= budget:
            break

        tau = (1.0 - rho) * tau

        # Update pheromones towards the top elite_frac proportion of ants
        q = max(1, int(np.ceil(elite_frac * X.shape[0])))
        idx = np.argsort(fitness)[-q:]

        # Adaptive deposit rate alpha: linear schedule from alpha0 -> alpha_min
        t = min(1.0, it / max(1, approx_iter_budget))
        alpha = alpha0 + (alpha_min - alpha0) * t

        # Rank weights (1..q), normalised
        ranks = np.arange(1, q + 1, dtype=float)
        w = ranks / ranks.sum()

        def deposit(tau_vec, x_bits, rate):
            return np.clip(tau_vec + rate * (x_bits - tau_vec), tau_min, tau_max)

        for j, wj in zip(idx, w):
            tau = deposit(tau, X[j], rate=alpha * wj)

        if use_best_so_far:
            tau = deposit(tau, best_x, rate=best_so_far_rate)

        # End of iteration
        history.append((evals, best_f))
        since_improv += 1

        if since_improv >= stall_window:
            ent = _bit_entropy(tau)
            ln2 = np.log(2.0)
            if ent < entropy_thresh * ln2:
                tau = (1 - restart_alpha) * tau + restart_alpha * 0.5
            since_improv = 0

        # Keeps fixed interval logs aligned even if there is no evaluation in this loop
        while evals >= next_log and next_log <= budget:
            history.append((next_log, best_f))
            next_log += log_interval

    if history[-1][0] < budget:
        history.append((budget, best_f))
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
        "entropy_thresh": entropy_thresh,
        "log_interval": log_interval,
        "best_solution": best_x.tolist(),
        "best_fitness": float(best_f),
        "history": history,
    }

    fname = os.path.join(outdir, f"aco_p{problem_id}_d{n}_s{seed}.json")
    with open(fname, "w") as f:
        json.dump(result, f, indent=2)

    return best_x, best_f
