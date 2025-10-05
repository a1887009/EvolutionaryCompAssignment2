import os
from genetic_algorithm import run_ga


def main():
    outdir = "data_ga"

    # only valid REAL problems
    problem_ids = [1, 2, 3, 18, 23, 24]
    dimension = 10
    budget = 10000
    seeds = [1, 2, 3, 4, 5]

    for pid in problem_ids:
        print(f"Running GA on F{pid} ...")
        for seed in seeds:
            run_ga(
                problem_id=pid,
                n=dimension,
                budget=budget,
                seed=seed,
                outdir=os.path.join(outdir, f"ga_f{pid}"),
            )
        print(f"Finished GA on F{pid}, results stored in {outdir}/ga_f{pid}/")


if __name__ == "__main__":
    main()
