from ioh import get_problem, ProblemClass, logger
import numpy as np


def randomized_local_search(func, budget=99999):
    n = func.meta_data.n_variables

    # special case handling
    if func.meta_data.problem_id == 18 and n == 32:
        optimum = 8
    else:
        optimum = func.optimum.y

    for run in range(10):  # 10 independent runs
        s = np.random.randint(2, size=n)
        f_s = func(s)
        f_best, s_best = f_s, s.copy()  # best-so-far

        for i in range(budget):
            # flip exactly one random bit
            s_prime = s.copy()
            j = np.random.randint(0, n)
            s_prime[j] = 1 - s_prime[j]

            f_prime = func(s_prime)
            if f_prime >= f_s:
                s, f_s = s_prime, f_prime
                if f_s > f_best:
                    f_best, s_best = f_s, s.copy()  # update best-so-far

            # early stop if optimum found
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
        folder_name="exercise2_RLS",
        algorithm_name="RLS",
        algorithm_info="Randomized Local Search with best-so-far logging"
    )

    for fid in fids:
        problem = get_problem(fid=fid, dimension=n, instance=instance, problem_class=ProblemClass.PBO)
        problem.attach_logger(l)
        randomized_local_search(problem)
        print(f"Completed RLS on F{fid}")

    del l


if __name__ == "__main__":
    main()
