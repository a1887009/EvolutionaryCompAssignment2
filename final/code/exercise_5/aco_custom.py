import os
import json
from datetime import datetime
from typing import Tuple, Optional

try:
    import ioh
except Exception:
    ioh = None

import numpy as np


def _get_pbo_problem(problem_id: int, n: int, instance: int = 1):
    try:
        return ioh.get_problem(problem_id, dimension=n, instance=instance, problem_type="PBO")
    except TypeError:
        pass
    except Exception:
        pass
    try:
        if hasattr(ioh, "ProblemClass"):
            pc = ioh.ProblemClass
            for attr in ("BOOLEAN", "PBO"):
                if hasattr(pc, attr):
                    return ioh.get_problem(problem_id, dimension=n, instance=instance, problem_class=getattr(pc, attr))
    except Exception:
        pass

    try:
        if hasattr(ioh, "ProblemType"):
            pt = ioh.ProblemType
            for attr in ("BOOLEAN", "BINARY", "PBO"):
                if hasattr(pt, attr):
                    return ioh.get_problem(problem_id, dimension=n, instance=instance, problem_type=getattr(pt, attr))
    except Exception:
        pass

    try:
        if hasattr(ioh, "suite") and hasattr(ioh.suite, "PBO"):
            try:
                suite = ioh.suite.PBO(functions=[problem_id], instances=[instance], dimensions=[n])
            except TypeError:
                suite = ioh.suite.PBO(problem_ids=[problem_id], instances=[instance], dimensions=[n])
            for prob in suite:
                return prob
    except Exception:
        pass

    raise RuntimeError(
        f"Unable to load PBO problem id={problem_id}. "
        "Tried problem_type, problem_class, ProblemType, and suite.PBO fallback. "
        "Consider upgrading 'ioh' to the latest version."
    )

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _evaluate(problem, x: np.ndarray) -> float:
    return float(problem(x))


def greedy_1bit_local_search(problem, x: np.ndarray) -> Tuple[np.ndarray, float, int]:
    """Best-improvement 1-bit flip; returns (x_best, f_best, evals_used)."""
    f0 = _evaluate(problem, x)
    n = x.size
    best_f = f0
    best_i = -1
    for i in range(n):
        x[i] ^= 1
        fi = _evaluate(problem, x)
        if fi > best_f:
            best_f, best_i = fi, i
        x[i] ^= 1
    evals_used = n
    if best_i >= 0:
        x[best_i] ^= 1
        return x, best_f, evals_used
    return x, f0, evals_used


def _rank_weights(mu: int):
    ranks = np.arange(1, mu + 1)
    w = np.log(mu + 1.0) - np.log(ranks)
    w /= w.sum()
    return w


def run_aco(
    problem_id: int,
    n: int = 100,
    budget: int = 100_000,
    seed: int = 1,
    outdir: str = ".",
    ants: int = 20,
    rho: float = 1.0 / 50.0,
    tau_min: Optional[float] = None,
    tau_max: Optional[float] = None,
    use_best_so_far: bool = True,
    use_local_search: bool = True,
):
    _ensure_dir(outdir)

    if ioh is None:
        raise RuntimeError("ioh is not available in this environment.")

    problem = _get_pbo_problem(problem_id, n, instance=1)

    rng = np.random.default_rng(seed)
    if tau_min is None:
        tau_min = 1.0 / (2.0 * n)
    if tau_max is None:
        tau_max = 1.0 - tau_min

    tau = np.full(n, 0.5, dtype=float)

    mu = max(2, ants // 5)
    w = _rank_weights(mu)
    bsf_weight = 0.30 * rho if use_best_so_far else 0.0

    evals = 0
    history = []
    best_x = (rng.random(n) < 0.5).astype(int)
    best_f = _evaluate(problem, best_x)
    evals += 1

    no_imp_limit = 200
    low_entropy_thr = 0.35
    no_imp_iters = 0

    def deposit(tau_vec, x_bits, amount):
        return np.clip(tau_vec + amount * (x_bits - tau_vec), tau_min, tau_max)

    X = np.empty((ants, n), dtype=int)
    F = np.empty(ants, dtype=float)

    while evals < budget:
        half = ants // 2
        U = rng.random((half, n))
        X[:half] = (U < tau).astype(int)
        X[half:2 * half] = ((1.0 - U) < tau).astype(int)
        if 2 * half < ants:
            X[-1] = (rng.random(n) < tau).astype(int)
        for a in range(ants):
            fx = _evaluate(problem, X[a])
            evals += 1
            if use_local_search:
                X[a], fx, used = greedy_1bit_local_search(problem, X[a])
                evals += used
            F[a] = fx
            if fx > best_f:
                best_f, best_x = fx, X[a].copy()
                no_imp_iters = 0
            if evals >= budget:
                break
        tau = (1.0 - rho) * tau
        elite_idx = np.argsort(-F)[:mu]
        for rank, idx in enumerate(elite_idx):
            tau = deposit(tau, X[idx], amount=rho * w[rank])

        if bsf_weight > 0.0:
            tau = deposit(tau, best_x, amount=bsf_weight)

        no_imp_iters += 1
        if no_imp_iters >= no_imp_limit:
            p = tau.clip(1e-9, 1 - 1e-9)
            H = -(p * np.log(p) + (1 - p) * np.log(1 - p))
            if H.mean() < low_entropy_thr:
                k = max(1, n // 10)
                idx = rng.choice(n, size=k, replace=False)
                tau[idx] = 0.5
            no_imp_iters = 0

        history.append((evals, float(best_f)))

    result = {
        "algorithm": "ACO-Custom-μbest-anti",
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
