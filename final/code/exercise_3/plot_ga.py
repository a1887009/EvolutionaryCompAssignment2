import os
import json
import numpy as np
import matplotlib.pyplot as plt

# function to plot results from multiple GA runs on specified problems
def plot_results(outdir="data_ga", problems=[1, 2, 3, 18, 23, 24]):
    for pid in problems:
        folder = os.path.join(outdir, f"ga_f{pid}")
        if not os.path.exists(folder):
            print(f"No results found for F{pid}")
            continue

        all_histories = []

        # load all histories for this problem
        for file in os.listdir(folder):
            if file.endswith(".json"):
                filepath = os.path.join(folder, file)
                with open(filepath, "r") as f:
                    data = json.load(f)

                if "history" not in data:
                    print(f"Skipping {filepath}, no history field")
                    continue # error handling

                history = data["history"]
                evals, fitness = zip(*history)
                all_histories.append(fitness)

        if not all_histories:
            print(f"No valid history data for F{pid}")
            continue

        # convert to numpy array for easy processing
        all_histories = np.array(all_histories)  # shape: (num_seeds, num_evals)
        evals = np.arange(1, all_histories.shape[1]+1)  # assumes all histories same length

        # compute mean and std deviation across seeds
        mean_fitness = np.mean(all_histories, axis=0)
        std_fitness = np.std(all_histories, axis=0)

        # plot individual seeds
        plt.figure(figsize=(8, 6))
        for seed_idx in range(all_histories.shape[0]):
            plt.plot(evals, all_histories[seed_idx], color='gray', alpha=0.3)

        # plot mean and std deviation
        plt.plot(evals, mean_fitness, color='blue', label="Mean fitness", linewidth=2)
        plt.fill_between(evals,
                         mean_fitness - std_fitness,
                         mean_fitness + std_fitness,
                         color='blue',
                         alpha=0.2,
                         label="±1 std dev")

        # formatting and axis titles
        plt.title(f"GA Convergence on F{pid}")
        plt.xlabel("Evaluations")
        plt.ylabel("Best Fitness")
        plt.yscale("linear") 
        plt.gca().invert_yaxis()  # lower fitness at top (otherwise would be downwards curve, purely aesthetic)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()

        outpath = os.path.join(outdir, f"ga_f{pid}_plot.pdf")
        plt.savefig(outpath, dpi=200)
        plt.close()
        print(f"Saved plot for F{pid} -> {outpath}")

# plot when run as script
if __name__ == "__main__":
    plot_results()
