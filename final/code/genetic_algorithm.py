import ioh
import numpy as np
import os

def run_ga(problem, budget, rng, pop_size=50, crossover_rate=0.9, mutation_rate=0.01, verbose=False):
    """
    Run a binary GA with uniform crossover and bit-flip mutation on a given IOH problem.
    Results are automatically logged to IOHanalyzer via the attached logger.
    """

    n = problem.meta_data.n_variables

    # initialise binary population
    population = rng.integers(0, 2, size=(pop_size, n))
    fitness = np.array([problem(ind) for ind in population])
    evaluations = pop_size

    best_fitness = np.max(fitness)

    while evaluations < budget:
        # tournament selection
        parents = []
        for _ in range(pop_size):
            i, j = rng.integers(0, pop_size, 2)
            if fitness[i] >= fitness[j]:
                parents.append(population[i])
            else:
                parents.append(population[j])
        parents = np.array(parents)

        # crossover (uniform)
        offspring = []
        for i in range(0, pop_size, 2):
            p1, p2 = parents[i], parents[min(i + 1, pop_size - 1)]
            if rng.random() < crossover_rate:
                mask = rng.integers(0, 2, n)  # random 0/1 mask
                c1 = np.where(mask, p1, p2)
                c2 = np.where(mask, p2, p1)
            else:
                c1, c2 = p1.copy(), p2.copy()
            offspring.append(c1)
            offspring.append(c2)
        offspring = np.array(offspring)

        # mutation (bit-flip)
        mutation_mask = rng.random(offspring.shape) < mutation_rate
        offspring = np.logical_xor(offspring, mutation_mask).astype(int)

        # evaluate offspring
        new_fitness = np.array([problem(ind) for ind in offspring])
        evaluations += len(offspring)

        # elitism: keep best from old population
        best_idx = np.argmax(fitness)
        worst_idx = np.argmin(new_fitness)
        offspring[worst_idx] = population[best_idx]
        new_fitness[worst_idx] = fitness[best_idx]

        # update population
        population = offspring
        fitness = new_fitness

        # track best
        gen_best = np.max(fitness)
        if gen_best > best_fitness:
            best_fitness = gen_best

        # optional progress printing
        if verbose and evaluations % (budget // 5) < pop_size:  # print ~5 updates per run
            print(f"    Evaluations: {evaluations}/{budget}, Best fitness so far: {best_fitness}")

    # return best solution found
    best_idx = np.argmax(fitness)
    return population[best_idx], fitness[best_idx]


def main():
    # GA experiment settings
    problems = [1, 2, 3, 18, 23, 24, 25]   # required PBO functions
    n = 100                                # problem size
    budget = 100_000                       # iterations/evaluations
    runs = 10                              # independent runs

    rng = np.random.default_rng(42)

    # fixed path for assignment
    root_dir = "/root/EVCOMP/EvolutionaryCompAssignment2/final/data/ex4/ioh_logs/GA"
    algo_name = "GeneticAlgorithm"

    os.makedirs(root_dir, exist_ok=True)

    for fid in problems:
        print(f"\n=== Running GA on Problem F{fid} (dimension={n}) ===")
        # setup IOH problem
        problem = ioh.get_problem(fid, dimension=n, instance=1, problem_class=ioh.ProblemClass.PBO)

        # create IOH logger (stores in GA/Fxx folders)
        l = ioh.logger.Analyzer(
            root=root_dir,
            folder_name=f"F{fid}",
            algorithm_name=algo_name,
            algorithm_info="Binary GA with uniform crossover and bit-flip mutation"
        )
        problem.attach_logger(l)

        # run GA multiple times
        for run in range(runs):
            print(f"  -> Run {run+1}/{runs}")
            seed = rng.integers(1e9)
            run_rng = np.random.default_rng(seed)
            problem.reset()
            best_sol, best_fit = run_ga(problem, budget, run_rng, verbose=True)
            print(f"     Finished Run {run+1}, Best fitness = {best_fit}")

        del l   # flush data to disk
        print(f"=== Finished all runs for Problem F{fid} ===\n")


if __name__ == "__main__":
    main()
