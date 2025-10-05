import json
import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt

# Aggregates 'history' from JSON logs and produces fixed-budget plots (mean ± 1 SD)
# Saves PDFs under: final/code/exercise_5/data/exercise5_F{pid}.pdf
# Reads run outputs from: final/code/exercise_5/data_aco/aco_f{pid}/*.json

def load_histories(pattern: str):
    """Load per-run histories [[evals, best], ...] from JSON files matching pattern."""
    histories = []
    for fp in glob(pattern):
        try:
            with open(fp, "r") as f:
                data = json.load(f)
            hist = data.get("history", [])
            if hist:
                histories.append(hist)
        except Exception:
            # skip unreadable/malformed files gracefully
            pass
    return histories

def resample(histories, budget: int, step: int = 1000):
    """
    Resample best-so-far curves onto a common evaluation grid.
    Returns (grid, mean, std). If no histories, (grid, None, None).
    """
    grid = np.arange(step, budget + 1, step)
    M = []
    for hist in histories:
        xs = np.array([h[0] for h in hist], dtype=float)
        ys = np.array([h[1] for h in hist], dtype=float)
        ys = np.maximum.accumulate(ys)  # enforce monotone best-so-far
        y_grid = np.interp(grid, xs, ys, left=ys[0], right=ys[-1])
        M.append(y_grid)
    if not M:
        return grid, None, None
    M = np.vstack(M)
    return grid, M.mean(axis=0), M.std(axis=0)

def fixed_budget_plot(pid: int, budget: int = 100000, step: int = 2000, outdir: str = "data"):
    """
    Build a fixed-budget plot for function pid.
    Outputs a PDF in final/code/exercise_5/data/
    """
    # Series to include (add MMAS/MMAS* later by exporting histories in same schema)
    series = {
        "ACO (yours)": f"data_aco/aco_f{pid}/*.json",
        # "MMAS": f"data_mmas/mmas_f{pid}/*.json",
        # "MMAS*": f"data_mmas_star/mmas_star_f{pid}/*.json",
    }

    plt.figure()
    for label, pattern in series.items():
        H = load_histories(pattern)
        grid, mu, sd = resample(H, budget=budget, step=step)
        if mu is None:
            continue
        # Plot mean and shaded ±1 SD; label both so legend is explicit
        plt.plot(grid, mu, label=f"{label} (mean)")
        plt.fill_between(grid, mu - sd, mu + sd, alpha=0.2, label=f"{label} (±1 SD)")

    plt.title(f"Fixed-budget performance on F{pid}")
    plt.xlabel("Evaluations")
    plt.ylabel("Best-so-far fitness (mean ± 1 SD)")
    plt.legend()
    os.makedirs(outdir, exist_ok=True)
    # Save as PDF into final/code/exercise_5/data/
    plt.savefig(os.path.join(outdir, f"exercise5_F{pid}.pdf"), format="pdf", bbox_inches="tight")
    plt.close()

def main():
    problems = [1, 2, 3, 18, 23, 24, 25]
    for pid in problems:
        fixed_budget_plot(pid)
    print("Saved PDF plots to final/code/exercise_5/data/")

if __name__ == "__main__":
    main()
