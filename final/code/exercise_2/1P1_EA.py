from ioh import get_problem, ProblemClass, logger
import sys
import numpy as np


def one_plus_one_EA(func, budget=99999):
    n = func.meta_data.n_variables

    if func.meta_data.problem_id == 18 and n == 32:
        optimum = 8
    else:
        optimum = func.optimum.y

    for run in range(10):  # 10 independent runs
        s = np.random.randint(2, size=n)
        f_s = func(s)
        f_opt, x_opt = f_s, s.copy()

        p = 1.0 / n
        for i in range(budget):
            # flip each bit with prob 1/n
            flips = np.random.rand(n) < p
            s_prime = s.copy()
            s_prime[flips] = 1 - s_prime[flips]

            f_prime = func(s_prime)
            if f_prime >= f_s:
                s, f_s = s_prime, f_prime
                if f_s > f_opt:
                    f_opt, x_opt = f_s, s.copy()

            if f_opt >= optimum:
                break

        func.reset()

    return f_opt, x_opt


def main():
    # problem set
    fids = [1, 2, 3, 18, 23, 24, 25]
    n = 100
    instance = 1

    # logger setup for IOHanalyzer
    l = logger.Analyzer(
        root="data",
        folder_name="exercise2_1P1_EA",
        algorithm_name="(1+1)EA",
        algorithm_info="(1+1) Evolutionary Algorithm implementation"
    )

    for fid in fids:
        problem = get_problem(fid=fid, dimension=n, instance=instance, problem_class=ProblemClass.PBO)
        problem.attach_logger(l)
        one_plus_one_EA(problem)
        print(f"Completed (1+1)EA on F{fid}")

    # ensure log flush
    del l


if __name__ == "__main__":
    main()
