import os
import numpy as np
from aco_custom import run_aco

def main():
    outdir = "data_aco"
    os.makedirs(outdir, exist_ok=True)

    problem_ids = [1, 2, 3, 18, 23, 24, 25]
    dimension = 100
    budget = 100_000

    ants = 24
    rho = 1.0 / 50
    tau_min, tau_max = None, None
    use_best_so_far = True
    use_local_search = True

    seeds = list(range(1, 11))

    for pid in problem_ids:
        print(f"[ACO] Running on F{pid} ...")
        scores = []
        for seed in seeds:
            _, best_f = run_aco(
                problem_id=pid,
                n=dimension,
                budget=budget,
                seed=seed,
                outdir=os.path.join(outdir, f"aco_f{pid}"),
                ants=ants,
                rho=rho,
                tau_min=tau_min,
                tau_max=tau_max,
                use_best_so_far=use_best_so_far,
                use_local_search=use_local_search,
            )
            scores.append(best_f)

        scores = np.array(scores, dtype=float)
        mean = scores.mean()
        std = scores.std(ddof=1) if len(scores) > 1 else 0.0
        best = scores.max()
        worst = scores.min()

        print(
            f"F{pid}: seeds={len(seeds)}  "
            f"mean={mean:.4f}  sd={std:.4f}  best={best:.4f}  worst={worst:.4f}"
        )

if __name__ == "__main__":
    main()
