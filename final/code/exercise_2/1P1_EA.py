from ioh import get_problem, ProblemClass, logger
import numpy as np


def one_plus_one_EA(func, budget=99999):
    n = func.meta_data.n_variables

    if func.meta_data.problem_id == 18 and n == 32:
        optimum = 8
    else:
        optimum = func.optimum.y

    p = 1.0 / n

    for run in range(10):
        s = np.random.randint(2, size=n)
        f_s = func(s)
        f_best, s_best = f_s, s.copy()  # best-so-far

        for i in range(budget):
            # flip each bit independently with probability 1/n
            flips = np.random.rand(n) < p
            s_prime = s.copy()
            s_prime[flips] = 1 - s_prime[flips]

            f_prime = func(s_prime)
            if f_prime >= f_s:
                s, f_s = s_prime, f_prime
                if f_s > f_best:
                    f_best, s_best = f_s, s.copy()  # update best-so-far

            if f_best >= optimum:
                break

        # Only reset between runs
        if run < 9:
            func.reset()

    return f_best, s_best


def main():
    fids = [1, 2, 3, 18, 23, 24, 25]
    n = 100
    instance = 1

    l = logger.Analyzer(
        root="data_e2",
        folder_name="exercise2_1P1_EA",
        algorithm_name="(1+1)EA",
        algorithm_info="(1+1) Evolutionary Algorithm with best-so-far logging"
    )

    for fid in fids:
        problem = get_problem(fid=fid, dimension=n, instance=instance, problem_class=ProblemClass.PBO)
        problem.attach_logger(l)
        one_plus_one_EA(problem)
        print(f"Completed (1+1)EA on F{fid}")

    del l


if __name__ == "__main__":
    main()
