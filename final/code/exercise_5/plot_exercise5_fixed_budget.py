import json
import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt

def load_histories(pattern: str):
    histories = []
    for fp in glob(pattern):
        try:
            with open(fp, "r") as f:
                data = json.load(f)
            hist = data.get("history", [])
            if hist:
                histories.append(hist)
        except Exception:
            pass
    return histories

def resample(histories, budget: int, step: int = 1):
    grid = np.arange(step, budget + 1, step)
    M = []
    for hist in histories:
        xs = np.array([h[0] for h in hist], dtype=float)
        ys = np.array([h[1] for h in hist], dtype=float)
        ys = np.maximum.accumulate(ys)
        y_grid = np.interp(grid, xs, ys, left=ys[0], right=ys[-1])
        M.append(y_grid)
    if not M:
        return grid, None, None
    M = np.vstack(M)
    return grid, M.mean(axis=0), M.std(axis=0)

def fixed_budget_plot(pid: int, budget: int = 200, step: int = 1, outdir: str = "data"):
    series = {
        "": f"data_aco/aco_f{pid}/*.json",
    }

    plt.figure()
    for label, pattern in series.items():
        H = load_histories(pattern)
        grid, mu, sd = resample(H, budget=budget, step=step)
        if mu is None:
            continue
        plt.plot(grid, mu, label=f"{label} mean fitness")
        plt.fill_between(grid, mu - sd, mu + sd, alpha=0.2, label=f"{label} ±1 std dev")

    plt.title(f"ACO Convergence on F{pid}")
    plt.xlabel("Evaluations")
    plt.ylabel("Best Fitness")
    plt.legend()
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(os.path.join(outdir, f"exercise5_F{pid}.pdf"), format="pdf", bbox_inches="tight")
    plt.close()

def main():
    problems = [1, 2, 3, 18, 23, 24, 25]
    for pid in problems:
        fixed_budget_plot(pid)
    print("Saved PDF plots to final/code/exercise_5/data/")

if __name__ == "__main__":
    main()
