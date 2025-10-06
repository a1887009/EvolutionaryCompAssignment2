import os
import json
from datetime import datetime
from typing import Tuple, Optional
import ioh
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
    raise RuntimeError(f"Unable to load PBO problem id={problem_id}. Tried multiple loaders.")

# Ensures the output directory exists 
def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

# Evaluates the problem with a given binary solution
def _evaluate(problem, x: np.ndarray) -> float:
    return float(problem(x))

 # Performs a greedy 1-bit local search to improve the solution of a binary problem
def greedy_1bit_local_search(problem, x: np.ndarray, max_evals: Optional[int] = None) -> Tuple[np.ndarray, float, int]:
    f0 = _evaluate(problem, x)
    n = x.size
    best_f = f0
    best_i = -1
    evals_used = 1
    budget_left = (max_evals - evals_used) if max_evals is not None else n

    # try flipping each bit and see if it improves the solution
    for i in range(n):
        if max_evals is not None and budget_left <= 0:
            break
        x[i] ^= 1
        fi = _evaluate(problem, x)
        evals_used += 1
        if max_evals is not None:
            budget_left -= 1
        if fi > best_f:
            best_f, best_i = fi, i
        x[i] ^= 1
    # if an improvement was found, apply it
    if best_i >= 0:
        x[best_i] ^= 1
        return x, best_f, evals_used
    return x, f0, evals_used

# Ranks weights for the top mu solutions in ACO based on logarithmic scaling 
def _rank_weights(mu: int):
    ranks = np.arange(1, mu + 1)
    w = np.log(mu + 1.0) - np.log(ranks)
    w /= w.sum()
    return w

# this computes the hamming distance between two binary vectors 
def _hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.sum(a ^ b))

# This selects a diverse set of elite solutions based on the hamming distance
def _select_diverse_elites(X: np.ndarray, F: np.ndarray, mu: int, d_min: int) -> np.ndarray:
    order = np.argsort(-F)
    selected = []
    for idx in order: # iterate in order of fitness
        if not selected:
            selected.append(idx)
        else:
            ok = True
            for j in selected: # checks if hamming distance to already selected elites is above threshold
                if _hamming(X[idx], X[j]) < d_min:
                    ok = False
                    break
            if ok:
                selected.append(idx)
        if len(selected) >= mu:
            break
    if len(selected) < mu:
        for idx in order:
            if idx not in selected:
                selected.append(idx)
                if len(selected) >= mu:
                    break
    return np.array(selected[:mu], dtype=int)

# this generates a Latin Hypercube Sample matrix for the bernoulli trials based on pheromone levels
def _lhs_bernoulli_matrix(tau: np.ndarray, ants: int, rng: np.random.Generator) -> np.ndarray:
    n = tau.size
    X = np.empty((ants, n), dtype=int)
    for j in range(n):
        strata = (np.arange(ants) + rng.random(ants)) / ants
        rng.shuffle(strata)
        X[:, j] = (strata < tau[j]).astype(int)
    return X

# The main ACO algorithm function
def run_aco(
    # parameters section
    problem_id: int,
    n: int = 100,
    budget: int = 100_000,
    seed: int = 1,
    outdir: str = ".",
    ants: int = 20, # number of ants per iteration
    rho: float = 1.0 / 50.0,
    tau_min: Optional[float] = None,
    tau_max: Optional[float] = None,
    use_best_so_far: bool = True,
    use_local_search: bool = True,
    elite_diversity: bool = True,
    elite_dmin_frac: float = 0.05,
    adaptive_rho: bool = True,
    sampling: str = "lhs",
    trim_frac: float = 0.2,
    max_step_frac: float = 0.08,
    entropy_adapt_mu: bool = True,
):
    # setup and checks
    _ensure_dir(outdir)
    # if ioh is not available, raise an error
    if ioh is None:
        raise RuntimeError("ioh not available")
    problem = _get_pbo_problem(problem_id, n, instance=1)
    rng = np.random.default_rng(seed)
    # parameter checks
    if tau_min is None:
        tau_min = 1.0 / (2.0 * n)
    if tau_max is None:
        tau_max = 1.0 - tau_min
    tau = np.full(n, 0.5, dtype=float)
    base_mu = max(2, ants // 5)
    w = _rank_weights(base_mu)
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
    d_min = max(1, int(elite_dmin_frac * n))
    step_cap = max_step_frac * (tau_max - tau_min)

    # while loop until the budget is exhausted
    # at each iteration, ants sample solutions are based on thepheromone levels
    while evals < budget:
        if sampling == "lhs":
            X = _lhs_bernoulli_matrix(tau, ants, rng)
        else:
            half = ants // 2
            U = rng.random((half, n))
            X[:half] = (U < tau).astype(int)
            X[half:2 * half] = ((1.0 - U) < tau).astype(int)
            if 2 * half < ants:
                X[-1] = (rng.random(n) < tau).astype(int)
        for a in range(ants): # this evaluates each ant's solution
            if evals >= budget:
                break
            fx = _evaluate(problem, X[a])
            evals += 1
            if use_local_search and evals < budget:
                max_ls = budget - evals
                if max_ls > 0:
                    X[a], fx, used = greedy_1bit_local_search(problem, X[a], max_evals=max_ls)
                    evals += (used - 1)
            F[a] = fx
            if fx > best_f:
                best_f, best_x = fx, X[a].copy()
                no_imp_iters = 0
        if evals >= budget:
            break
        rho_eff = rho
        if adaptive_rho:
            meanF = float(np.mean(F))
            stdF = float(np.std(F)) + 1e-12
            cv = stdF / (abs(meanF) + 1e-12)
            scale = 1.0 / (1.0 + cv)
            rho_eff = float(np.clip(rho * scale, 0.25 * rho, 1.25 * rho))
        tau_evap = (1.0 - rho_eff) * tau
        if entropy_adapt_mu:
            p = tau.clip(1e-9, 1 - 1e-9)
            H = -(p * np.log(p) + (1 - p) * np.log(1 - p))
            mu_eff = int(np.clip(round(base_mu * (1.0 + (0.5 - min(0.5, float(H.mean()))))), 2, ants))
            w = _rank_weights(mu_eff)
        else:
            mu_eff = base_mu
        if elite_diversity:
            elite_idx_full = _select_diverse_elites(X, F, mu=mu_eff, d_min=d_min)
        else:
            elite_idx_full = np.argsort(-F)[:mu_eff]
        k = max(2, int(np.ceil(mu_eff * (1.0 - trim_frac))))
        elite_idx = elite_idx_full[:k]
        tau_work = tau_evap.copy()
        for rank, idx in enumerate(elite_idx): # this deposits the pheromone from elite solutions
            tau_work = deposit(tau_work, X[idx], amount=rho_eff * w[rank if rank < w.size else -1])
        if bsf_weight > 0.0:
            tau_work = deposit(tau_work, best_x, amount=bsf_weight)
        diff = tau_work - tau_evap
        diff = np.clip(diff, -step_cap, step_cap)
        tau = np.clip(tau_evap + diff, tau_min, tau_max)
        no_imp_iters += 1
        if no_imp_iters >= no_imp_limit:
            p = tau.clip(1e-9, 1 - 1e-9)
            H = -(p * np.log(p) + (1 - p) * np.log(1 - p))
            if H.mean() < low_entropy_thr:
                kreset = max(1, n // 10)
                idx = rng.choice(n, size=kreset, replace=False)
                tau[idx] = 0.5
            no_imp_iters = 0
        history.append((evals, float(best_f)))
    # this section saves the result to a json file
    result = {
        "algorithm": "ACO-VarianceReduced(LHS+Trim+Cap)+diversity+adaptive",
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
        "elite_diversity": elite_diversity,
        "elite_dmin_frac": elite_dmin_frac,
        "adaptive_rho": adaptive_rho,
        "sampling": sampling,
        "trim_frac": trim_frac,
        "max_step_frac": max_step_frac,
        "entropy_adapt_mu": entropy_adapt_mu,
        "best_solution": best_x.tolist(),
        "best_fitness": float(best_f),
        "history": history,
    }
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = os.path.join(outdir, f"aco_p{problem_id}_d{n}_s{seed}_{timestamp}.json")
    with open(fname, "w") as f:
        json.dump(result, f, indent=2)
    return best_x, best_f
