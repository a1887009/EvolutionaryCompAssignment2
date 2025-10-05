import os
import json
import matplotlib.pyplot as plt


def plot_results(outdir="data_ga", problems=[1, 2, 3, 18, 23, 24]):
    for pid in problems:
        plt.figure(figsize=(8, 6))

        folder = os.path.join(outdir, f"ga_f{pid}")
        if not os.path.exists(folder):
            print(f"No results found for F{pid}")
            continue

        has_data = False
        for file in os.listdir(folder):
            if file.endswith(".json"):
                filepath = os.path.join(folder, file)
                with open(filepath, "r") as f:
                    data = json.load(f)

                if "history" not in data:
                    print(f"Skipping {filepath}, no history field")
                    continue

                history = data["history"]
                evals, fitness = zip(*history)
                plt.plot(evals, fitness, alpha=0.7, label=f"seed={data['seed']}")
                has_data = True

        if has_data:
            plt.title(f"GA Convergence on F{pid}")
            plt.xlabel("Evaluations")
            plt.ylabel("Best Fitness")
            plt.yscale("log")  # optional
            plt.gca().invert_yaxis()  # flip y-axis so lower fitness is at top
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.tight_layout()

            outpath = os.path.join(outdir, f"ga_f{pid}_plot.pdf")
            plt.savefig(outpath, dpi=200)
            plt.close()
            print(f"Saved plot for F{pid} -> {outpath}")
        else:
            print(f"No valid history data for F{pid}")


if __name__ == "__main__":
    plot_results()
