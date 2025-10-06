import ioh
import numpy as np
import os
import json
from datetime import datetime


def run_ga(problem_id, n, budget, seed, outdir):
    # setup ioh problem with defined id and dimension
    problem = ioh.get_problem(problem_id, dimension=n, instance=1)

    # randomly generates seeded numbers for reproducibility
    rng = np.random.default_rng(seed)

    # ga parameters
    pop_size = 50
    mutation_rate = 0.1
    crossover_rate = 0.9

    # initialise population uniformly in boundary [-5, 5]
    population = rng.uniform(-5, 5, size=(pop_size, n))
    fitness = np.array([problem(ind) for ind in population])

    best_solution = population[np.argmin(fitness)] # best solution in initial pop
    best_fitness = np.min(fitness) # best fitness in initial pop

    evaluations = pop_size 
    history = [(evaluations, best_fitness)]

    # while loop until budget exhausted
    while evaluations < budget:
        # parent selection (tournament)
        parents = []
        for _ in range(pop_size):
            i, j = rng.integers(0, pop_size, 2)
            if fitness[i] < fitness[j]:
                parents.append(population[i])
            else:
                parents.append(population[j])
        parents = np.array(parents)

        # crossover to produce offspring
        offspring = []
        for i in range(0, pop_size, 2):
            p1, p2 = parents[i], parents[min(i + 1, pop_size - 1)]
            if rng.random() < crossover_rate:
                alpha = rng.random()
                c1 = alpha * p1 + (1 - alpha) * p2 # linear crossover
                c2 = alpha * p2 + (1 - alpha) * p1 
            else:
                c1, c2 = p1, p2
            offspring.append(c1) # append both children
            offspring.append(c2)
        offspring = np.array(offspring) # convert to numpy array

        # mutation to introduce diversity
        mutation_mask = rng.random(offspring.shape) < mutation_rate
        offspring = offspring + mutation_mask * rng.normal(0, 0.1, offspring.shape)

        # bound offspring within [-5, 5]
        offspring = np.clip(offspring, -5, 5)

        # evaluate best offspring
        new_fitness = np.array([problem(ind) for ind in offspring])
        evaluations += len(offspring)

        # elitism (keep best from previous generation)
        best_idx = np.argmin(fitness)
        worst_idx = np.argmax(new_fitness)
        offspring[worst_idx] = population[best_idx]
        new_fitness[worst_idx] = fitness[best_idx]

        # update population with best offspring
        population = offspring
        fitness = new_fitness

        # track best
        gen_best_idx = np.argmin(fitness)
        gen_best_fit = fitness[gen_best_idx]
        if gen_best_fit < best_fitness:
            best_solution = population[gen_best_idx]
            best_fitness = gen_best_fit

        history.append((evaluations, best_fitness))

    # save results
    os.makedirs(outdir, exist_ok=True)
    result = {
        "problem_id": problem_id,
        "dimension": n,
        "budget": budget,
        "seed": seed,
        "best_solution": best_solution.tolist(),
        "best_fitness": float(best_fitness),
        "history": history,
    }
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(outdir, f"ga_p{problem_id}_d{n}_s{seed}_{timestamp}.json")
    with open(filename, "w") as f:
        json.dump(result, f, indent=2)

    return best_solution, best_fitness
