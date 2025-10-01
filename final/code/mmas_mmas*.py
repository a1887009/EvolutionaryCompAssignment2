from ioh import get_problem, ProblemClass, logger
import numpy as np
import math
import sys
import random

def run_mmas(func, budget=100000, rho=1.0, use_mmas_star=False):

    n = func.meta_data.n_variables

    tau_min = 1.0 / (2*n)
    tau_max = 1.0 - tau_min

    pheromones = np.full(n, 0.5)

    if func.meta_data.problem_id == 18 and n == 32:
        optimum = 8
    else:
        optimum = func.optimum.y

    for run in range(10):
        best_solution = None
        best_fitness = -sys.float_info.max

        for t in range(budget):
            solution = np.array([1 if random.random() < p else 0 for p in pheromones])
            fitness = func(solution)

            if fitness > best_fitness:
                best_solution = solution.copy()
                best_fitness = fitness

            if best_fitness >= optimum:
                break

            pheromones = (1 - rho) * pheromones

            chosen = solution if use_mmas_star else best_solution
            for i in range(n):
                if chosen[i] == 1:
                    pheromones[i] += rho
                pheromones[i] = min(max(pheromones[i], tau_min), tau_max)

        func.reset()

    return best_fitness, best_solution

if __name__ == "__main__":
    n = 100
    budget = 100000

    problems = [
        get_problem(fid=1, dimension=n, instance=1, problem_class=ProblemClass.PBO),
        get_problem(fid=2, dimension=n, instance=1, problem_class=ProblemClass.PBO),
        get_problem(fid=3, dimension=n, instance=1, problem_class=ProblemClass.PBO),
        get_problem(fid=18, dimension=n, instance=1, problem_class=ProblemClass.PBO),
        get_problem(fid=23, dimension=n, instance=1, problem_class=ProblemClass.PBO),
        get_problem(fid=24, dimension=n, instance=1, problem_class=ProblemClass.PBO),
        get_problem(fid=25, dimension=n, instance=1, problem_class=ProblemClass.PBO),
    ]

    rhos = [1.0, 1.0/math.sqrt(n), 1.0/n]

    l = logger.Analyzer(root="data",
        folder_name="section4",
        algorithm_name="MMAS_vs_MMAS*",
        algorithm_info="Comparison of MMAS and MMAS*")
    
    for problem in problems:
        problem.attach_logger(l)

        for rho in rhos:
            run_mmas(problem, budget=budget, rho=rho, use_mmas_star=False)

            run_mmas(problem, budget=budget, rho=rho, use_mmas_star=True)

    del l