import json
import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt

# Aggregates 'history' from JSON logs and produces fixed-budget plots (mean ± 1 SD)
# Outputs figures as PDF files in final/doc/exercise_5/

def load_histories(pattern):
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

def resample(histories, budget, step=1000):
    """Resample best-so-far curves at common evaluation steps."""
    grid = np.arange(step, budget + 1, step)
    M = []
    for hist in histories:
        xs = np.array([h[0] for h in hist], dtype=float)
        ys = np.array([h[1] for h in hist], dtype=float)
        ys = np.maximum.accumulate(ys)  # best-so-far monotone
        y_grid = np.interp(grid, xs, ys, left=ys[0], right=ys[-1])
        M.append(y_grid)
    if not M:
        return grid, None, None
    M = np.vstack(M)
    return grid, M.mean(axis=0), M.std(axis=0)

def fixed_budget_plot(pid, budget=100000, step=2000, outdir="../doc/exercise_5"):
    series = {
        "ACO (yours)": f"data_aco/aco_f{pid}/*.json",
        # Uncomment to include these when you export histories similarly:
        # "MMAS": f"data_mmas/mmas_f{pid}/*.json",
        # "MMAS*": f"data_mmas_star/mmas_star_f{pid}/*.json",
    }

    plt.figure()
    for label, pattern in series.items():
        H = load_histories(pattern)
        grid, mu, sd = resample(H, budget=budget, step=step)
        if mu is None:
            continue
        # Plot mean and shaded ±1 SD, both visible in legend
        mean_line, = plt.plot(grid, mu, label=f"{label} (mean)")
        plt.fill_between(grid, mu - sd, mu + sd, alpha=0.2, label=f"{label} (±1 SD)")

    plt.title(f"Fixed-budget performance on F{pid}")
    plt.xlabel("Evaluations")
    plt.ylabel("Best-so-far fitness (mean ± 1 SD)")
    plt.legend()
    os.makedirs(outdir, exist_ok=True)
    # SAVE AS PDF
    plt.savefig(os.path.join(outdir, f"exercise5_F{pid}.pdf"), format="pdf", bbox_inches="tight")
    plt.close()

def main():
    problems = [1, 2, 3, 18, 23, 24, 25]
    for pid in problems:
        fixed_budget_plot(pid)
    print("Saved PDF plots to final/doc/exercise_5/")

if __name__ == "__main__":
    main()
