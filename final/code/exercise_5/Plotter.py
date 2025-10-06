import os
import json
import numpy as np
import matplotlib.pyplot as plt

def _load_histories(folder):
    runs = []
    for file in os.listdir(folder):
        if file.endswith(".json"):
            with open(os.path.join(folder, file), "r") as f:
                data = json.load(f)
            if "history" in data and len(data["history"]) > 0:
                ev, fit = zip(*data["history"])
                runs.append((np.array(ev, dtype=int), np.array(fit, dtype=float)))
    return runs

def _align_on_grid(runs):
    grid = np.unique(np.concatenate([ev for ev, _ in runs]))
    aligned = []
    for ev, fit in runs:
        idx = np.searchsorted(ev, grid, side="right") - 1
        idx[idx < 0] = 0
        fit_on_grid = fit[idx]
        aligned.append(fit_on_grid)
    return grid, np.vstack(aligned)

# Main plotting function
def plot_aco_results(outdir="data_aco", problems=[1, 2, 3, 18, 23, 24, 25]):
    # this ensures output directory exists for plots
    for pid in problems:
        folder = os.path.join(outdir, f"aco_f{pid}")
        if not os.path.exists(folder):
            print(f"No results found for F{pid}")
            continue

        runs = _load_histories(folder)
        # if no valid runs, skip
        if not runs:
            print(f"No valid history data for F{pid}")
            continue

        evals, M = _align_on_grid(runs)
        mean_fitness = M.mean(axis=0)
        std_fitness = M.std(axis=0)

        plt.figure(figsize=(8, 6))
        for r in range(M.shape[0]):
            plt.plot(evals, M[r], color="gray", alpha=0.3)

        plt.plot(evals, mean_fitness, color="blue", label="Mean fitness", linewidth=2)
        plt.fill_between(
            evals,
            mean_fitness - std_fitness,
            mean_fitness + std_fitness,
            color="blue",
            alpha=0.2,
            label="±1 std dev"
        )

        plt.title(f"ACO Convergence on F{pid}")
        plt.xlabel("Evaluations")
        plt.ylabel("Best Fitness")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()

        outpath = os.path.join(outdir, f"aco_f{pid}_plot.pdf")
        plt.savefig(outpath, dpi=200)
        plt.close()
        print(f"Saved plot for F{pid} -> {outpath}")

if __name__ == "__main__":
    plot_aco_results()
