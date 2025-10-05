import os
import json
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = "/root/EVCOMP/EvolutionaryCompAssignment2/final/data/ex4/ioh_logs/GA"

def plot_results():
    for func_folder in sorted(os.listdir(DATA_DIR)):
        func_path = os.path.join(DATA_DIR, func_folder)
        if not os.path.isdir(func_path):
            continue

        json_files = [f for f in os.listdir(func_path) if f.endswith(".json")]
        if not json_files:
            print(f"⚠️ No JSON file in {func_folder}")
            continue

        json_path = os.path.join(func_path, json_files[0])
        try:
            with open(json_path, "r") as f:
                data = json.load(f)

            func_name = data.get("function_name", func_folder)
            runs = data.get("scenarios", [])[0].get("runs", [])
            if not runs:
                print(f"⚠️ No runs in {func_name}")
                continue

            best_y = [run["best"]["y"] for run in runs if "best" in run]
            if not best_y:
                print(f"⚠️ No best values for {func_name}")
                continue

            # calculate stats
            mean_y = np.mean(best_y)
            std_y = np.std(best_y)

            # plot results per run
            plt.figure(figsize=(8, 5))
            plt.plot(range(1, len(best_y) + 1), best_y, marker="o", linestyle="-", label="Best per run")
            plt.axhline(mean_y, color="red", linestyle="--", label=f"Mean = {mean_y:.2f}")
            plt.fill_between(
                range(1, len(best_y) + 1),
                mean_y - std_y,
                mean_y + std_y,
                color="red",
                alpha=0.2,
                label=f"±1 Std = {std_y:.2f}"
            )
            plt.title(f"GA Results on {func_name}")
            plt.xlabel("Run Index")
            plt.ylabel("Best Fitness Value")
            plt.legend()
            plt.tight_layout()

            out_path = os.path.join(func_path, f"{func_name}_GA.pdf")
            plt.savefig(out_path)
            plt.close()
            print(f"✅ Saved plot for {func_name} → {out_path}")

        except Exception as e:
            print(f"⚠️ Error processing {json_path}: {e}")


if __name__ == "__main__":
    plot_results()
