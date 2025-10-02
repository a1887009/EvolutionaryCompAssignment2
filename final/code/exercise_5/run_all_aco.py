import os
from aco_custom import run_aco

def main():
    outdir = "data_aco"
    os.makedirs(outdir, exist_ok=True)

    # Assignment settings
    problem_ids = [1, 2, 3, 18, 23, 24, 25]  # PBO functions
    dimension = 100
    budget = 100000
    seeds = list(range(1, 11))  # 10 independent runs

    # ACO hyperparameters (feel free to tune)
    ants = 20
    rho = 1.0 / 50  # evaporation
    tau_min, tau_max = 0.01, 0.99
    use_best_so_far = True
    use_local_search = True

    for pid in problem_ids:
        print(f"[ACO] Running on F{pid} ...")
        alg_dir = os.path.join(outdir, f"aco_f{pid}")
        os.makedirs(alg_dir, exist_ok=True)
        for seed in seeds:
            run_aco(
                problem_id=pid,
                n=dimension,
                budget=budget,
                seed=seed,
                outdir=alg_dir,
                ants=ants,
                rho=rho,
                tau_min=tau_min,
                tau_max=tau_max,
                use_best_so_far=use_best_so_far,
                use_local_search=use_local_search,
            )
        print(f"[ACO] Finished F{pid}. Results in {alg_dir}")

if __name__ == "__main__":
    main()
