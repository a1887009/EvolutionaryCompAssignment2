from ioh import get_problem, ProblemClass, logger
import sys
import numpy as np


def randomized_local_search(func, budget=99999):
    n = func.meta_data.n_variables

    # handle known special case from example
    if func.meta_data.problem_id == 18 and n == 32:
        optimum = 8
    else:
        optimum = func.optimum.y

    for run in range(10):  # 10 independent runs
        # initial solution
        s = np.random.randint(2, size=n)
        f_s = func(s)
        f_opt, x_opt = f_s, s.copy()

        for i in range(budget):
            # flip exactly one random bit
            s_prime = s.copy()
            j = np.random.randint(0, n)
            s_prime[j] = 1 - s_prime[j]

            f_prime = func(s_prime)
            if f_prime >= f_s:
                s, f_s = s_prime, f_prime
                if f_s > f_opt:
                    f_opt, x_opt = f_s, s.copy()

            # early stop if optimum found
            if f_opt >= optimum:
                break

        func.reset()  # required between runs

    return f_opt, x_opt


def main():
    # problem set
    fids = [1, 2, 3, 18, 23, 24, 25]
    n = 100
    instance = 1

    # logger setup for IOHanalyzer
    l = logger.Analyzer(
        root="data",
        folder_name="exercise2_RLS",
        algorithm_name="RLS",
        algorithm_info="Randomized Local Search implementation"
    )

    for fid in fids:
        problem = get_problem(fid=fid, dimension=n, instance=instance, problem_class=ProblemClass.PBO)
        problem.attach_logger(l)
        randomized_local_search(problem)
        print(f"Completed RLS on F{fid}")

    # ensure log flush
    del l


if __name__ == "__main__":
    main()
