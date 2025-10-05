from __future__ import annotations
import argparse
import math
import os
import sys
from dataclasses import dataclass
from typing import Literal, Tuple, Optional

import numpy as np

try:
    from ioh import get_problem, ProblemClass, logger
except Exception as e:
    print("[FATAL] Could not import 'ioh'. Please install IOHexperimenter: pip install ioh", file=sys.stderr)
    raise

AlgorithmName = Literal["MMAS", "MMAS_STAR"]

@dataclass
class Config:
    n: int = 100
    budget: int = 100_000
    runs: int = 10
    functions: Tuple[int, ...] = (1, 2, 3, 18, 23, 24, 25)
    algos: Tuple[AlgorithmName, ...] = ("MMAS", "MMAS_STAR")
    seed: Optional[int] = None
    root: str = "final/data/ex4/ioh_logs"

    @property
    def rhos(self):
        return (1.0, 1.0 / math.sqrt(self.n), 1.0 / self.n)


# ---------- MMAS core ----------
def construct_solution(pheromones: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return (rng.random(size=pheromones.size) < pheromones).astype(np.int8)


def clamp_pheromones(tau: np.ndarray, tau_min: float, tau_max: float) -> None:
    np.clip(tau, tau_min, tau_max, out=tau)


def mmas_update(
    pheromones: np.ndarray,
    chosen_bits: np.ndarray,
    rho: float,
    tau_min: float,
    tau_max: float,
) -> None:
    
    pheromones *= (1.0 - rho)
    #Add ρ only where chosen bit is 1
    pheromones[chosen_bits == 1] += rho
    clamp_pheromones(pheromones, tau_min, tau_max)


def run_mmas(
    func,
    budget: int,
    rho: float,
    use_mmas_star: bool,
    runs: int = 10,
    seed: int = 0,
) -> Tuple[float, np.ndarray]:
  
    n = func.meta_data.n_variables
    #Classic bounds for binary MMAS
    tau_min = 1.0 / (2 * n)
    tau_max = 1.0 - tau_min

    global_best_sol: Optional[np.ndarray] = None
    global_best_fit = -sys.float_info.max

    #If known, use optimum for early stopping
    try:
        known_optimum = float(func.optimum.y)
        if not np.isfinite(known_optimum):
            known_optimum = float("inf")
    except Exception:
        known_optimum = float("inf")

    base_rng = np.random.default_rng(seed)

    for run_idx in range(runs):
        rng = np.random.default_rng(base_rng.integers(0, 2**32 - 1))

        pheromones = np.full(n, 0.5, dtype=float)
        best_solution: Optional[np.ndarray] = None
        best_fitness = -sys.float_info.max

        for t in range(budget):
            sol = construct_solution(pheromones, rng)
            fit = func(sol)  # <-- IOH logger records this call automatically

            #Track run-best
            if fit > best_fitness:
                best_fitness = fit
                best_solution = sol.copy()

            #Choose reinforcement solution based on variant
            #MMAS -> iteration-best (current solution), MMAS* -> best-so-far in the run
            chosen = sol if (not use_mmas_star) else (best_solution if best_solution is not None else sol)
            mmas_update(pheromones, chosen, rho, tau_min, tau_max)

            #Early stop if optimum reached (optional)
            if best_fitness >= known_optimum:
                break

        #Mark end of an independent run in IOH by resetting the problem state
        try:
            func.reset()
        except Exception:
            pass

        if best_fitness > global_best_fit:
            global_best_fit = best_fitness
            global_best_sol = best_solution.copy() if best_solution is not None else None

    return global_best_fit, (global_best_sol if global_best_sol is not None else np.zeros(n, dtype=np.int8))


#Logging helpers
def make_logger(root: str, folder: str, algo_name: str, algo_info: str):

    return logger.Analyzer(
        root=root,
        folder_name=folder,
        algorithm_name=algo_name,
        algorithm_info=algo_info,
    )


def robust_attach(problem, L) -> None:
    attached = False
    try:
        L.attach_problem(problem)
        attached = True
    except Exception:
        try:
            problem.attach_logger(L)
            attached = True
        except Exception:
            pass
    if not attached:
        raise RuntimeError("Failed to attach IOH logger to the problem. Check your ioh version.")


def robust_detach(problem, L) -> None:
    try:
        problem.detach_logger()
    except Exception:
        pass
    try:
        L.close()
    except Exception:
        pass


def main():
    p = argparse.ArgumentParser(description="Exercise 4 runner: MMAS and MMAS* with IOH logger")
    p.add_argument("--n", type=int, default=100, help="Problem dimension (default: 100)")
    p.add_argument("--budget", type=int, default=100_000, help="Evaluations per run (default: 100000)")
    p.add_argument("--runs", type=int, default=10, help="Independent runs per setting (default: 10)")
    p.add_argument("--seed", type=int, default=None, help="Base RNG seed (default: None)")
    p.add_argument("--root", type=str, default="final/data/ex4/ioh_logs", help="Root folder for IOH logs")
    p.add_argument("--functions", type=int, nargs="*", default=[1, 2, 3, 18, 23, 24, 25], help="Function IDs to run")
    args = p.parse_args()

    cfg = Config(
        n=args.n,
        budget=args.budget,
        runs=args.runs,
        functions=tuple(args.functions),
        seed=args.seed,
        root=args.root,
    )

    os.makedirs(cfg.root, exist_ok=True)

    #Build the problem list up-front (PBO class)
    problems = [
        get_problem(fid=fid, dimension=cfg.n, instance=1, problem_class=ProblemClass.PBO)
        for fid in cfg.functions
    ]

    print(f"Running functions {list(cfg.functions)} with rhos {list(cfg.rhos)}; logs -> {cfg.root}")

    for problem in problems:
        fid = problem.meta_data.problem_id

        for rho in cfg.rhos:
            rho_tag = f"rho_{rho:.6f}"

            #MMAS (iteration-best)
            mmas_folder = f"MMAS/{rho_tag}/F{fid}"
            mmas_name   = f"MMAS_fid={fid}_n={cfg.n}"
            mmas_info   = "MMAS (iteration-best reinforcement)"
            L = make_logger(cfg.root, mmas_folder, mmas_name, mmas_info)
            robust_attach(problem, L)
            print(f"[RUN] F{fid} | MMAS | {rho_tag} | runs={cfg.runs} | budget={cfg.budget}")
            run_mmas(problem, budget=cfg.budget, rho=rho, use_mmas_star=False, runs=cfg.runs, seed=(cfg.seed or 0) + fid)
            robust_detach(problem, L)

            #MMAS* (best-so-far)
            mmas_star_folder = f"MMAS_STAR/{rho_tag}/F{fid}"
            mmas_star_name   = f"MMAS*_fid={fid}_n={cfg.n}"
            mmas_star_info   = "MMAS* (best-so-far reinforcement)"
            L = make_logger(cfg.root, mmas_star_folder, mmas_star_name, mmas_star_info)
            robust_attach(problem, L)
            print(f"[RUN] F{fid} | MMAS* | {rho_tag} | runs={cfg.runs} | budget={cfg.budget}")
            run_mmas(problem, budget=cfg.budget, rho=rho, use_mmas_star=True, runs=cfg.runs, seed=(cfg.seed or 2025) + fid)
            robust_detach(problem, L)

    print("All runs finished. Logs are under:", cfg.root)
    print("Open the root in IOHanalyzer to generate fixed-budget plots.")


if __name__ == "__main__":
    main()

