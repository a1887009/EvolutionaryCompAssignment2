import json
import os
from glob import glob
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

# This script aggregates 'history' from GA / ACO / MMAS JSON logs in final/code/*/ directories
# and produces fixed-budget plots (mean ± std) per function in final/doc/exercise_5/.
#
# Expected input directories (relative to script working dir = final/code):
#   - data_aco/aco_f{pid}/*.json          (created by run_all_aco.py)
#   - data_mmas/mmas_f{pid}/*.json        (if you export MMAS runs similarly)
#   - data_mmas_star/mmas_star_f{pid}/*.json
#
# You can adapt the patterns below to match your file layout.

def load_histories(pattern):
    histories = []
    for fp in glob(pattern):
        with open(fp, "r") as f:
            try:
                data = json.load(f)
                hist = data.get("history", [])
                histories.append(hist)
            except Exception:
                pass
    return histories

def resample(histories, budget, step=1000):
    """Resample best-so-far curves at common evaluation steps."""
    grid = np.arange(step, budget + 1, step)
    M = []
    for hist in histories:
        # hist is list of [evals, best]
        xs = np.array([h[0] for h in hist], dtype=float)
        ys = np.array([h[1] for h in hist], dtype=float)
        # best-so-far monotone
        ys = np.maximum.accumulate(ys)
        # interpolate
        y_grid = np.interp(grid, xs, ys, left=ys[0], right=ys[-1])
        M.append(y_grid)
    if not M:
        return grid, None, None
    M = np.vstack(M)
    return grid, M.mean(axis=0), M.std(axis=0)

def fixed_budget_plot(pid, budget=100000, step=2000, outdir="../doc/exercise_5"):
    series = {
        "ACO (yours)": f"data_aco/aco_f{pid}/*.json",
        # Uncomment / adjust these when you have exported MMAS & MMAS* histories:
        # "MMAS": f"data_mmas/mmas_f{pid}/*.json",
        # "MMAS*": f"data_mmas_star/mmas_star_f{pid}/*.json",
    }

    plt.figure()
    for label, pattern in series.items():
        H = load_histories(pattern)
        grid, mu, sd = resample(H, budget=budget, step=step)
        if mu is None:
            continue
        plt.plot(grid, mu, label=label)
        plt.fill_between(grid, mu - sd, mu + sd, alpha=0.2)
    plt.title(f"Fixed-budget performance on F{pid}")
    plt.xlabel("Evaluations")
    plt.ylabel("Best-so-far fitness")
    plt.legend()
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(os.path.join(outdir, f"exercise5_F{pid}.png"), dpi=180, bbox_inches="tight")
    plt.close()

def main():
    problems = [1,2,3,18,23,24,25]
    for pid in problems:
        fixed_budget_plot(pid)
    print("Saved plots to final/doc/exercise_5/")

if __name__ == "__main__":
    main()
