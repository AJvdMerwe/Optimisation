__author__ = 'Arneau Jacques van der Merwe'

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from logger_setup.logger import logger


def geometric_cooling_schedule(beta: object, t: object) -> object:
    return beta * t


def geometric_reheating_schedule(alpha: object, t: object) -> object:
    return alpha * t


def linear_cooling_schedule(beta: object, t: object) -> object:
    return t - beta


def linear_reheating_schedule(alpha: object, t: object) -> object:
    return t + alpha


def f_1(x: object) -> object:
    return np.power(x, 2)


def f_2(x: object) -> object:
    return np.power((x - 2), 2)


def histogram_diversity_preservation(f_x: object, archive: object, archive_threshold_length: object) -> object:
    """
    Generate cut points based off x,y ranges, split into 2,find ranges for for cutpoints. place f_x in a quadrant.
    Count number of elements within quadrants. Reject if number of solutions in quadrant is above
    mean number per quadrant
    :param f_x: candidate solution
    :param archive: archive of solution for possible pareto front approximation
    :param archive_threshold_length: minimum number for archive elements before diversity preservation is implemented
    :return: flag if candidate solution should be removed or not
    """

    x_1_s = []
    x_2_s = []
    add_bool = True
    temp_arch = archive + [f_x]
    for i in temp_arch:
        x_1_s.append(i[0])
        x_2_s.append(i[1])
    if len(temp_arch) > archive_threshold_length:
        q_1_count = 0
        q_2_count = 0
        q_3_count = 0
        q_4_count = 0

        x_1_s_median = np.median(np.array(x_1_s))  # generate cut point for F_1
        x_2_s_median = np.median(np.array(x_2_s))  # generate cut point for F_2

        # count number of candidate solution in each quadrant
        for a in archive:
            if (a[0] < x_1_s_median) and (a[1] < x_2_s_median):
                q_1_count += 1
            if (a[0] > x_1_s_median) and (a[1] > x_2_s_median):
                q_4_count += 1
            if (a[0] > x_1_s_median) and (a[1] < x_2_s_median):
                q_2_count += 1
            if (a[0] < x_1_s_median) and (a[1] > x_2_s_median):
                q_3_count += 1

        # mean number of solution in search space
        quadrant_mean = np.array([q_1_count, q_2_count, q_3_count, q_4_count]).mean()
        if (f_x[0] < x_1_s_median) and (f_x[1] < x_2_s_median):
            if q_1_count > quadrant_mean:
                add_bool = False  # candidate solution should be rejected
        if (f_x[0] > x_1_s_median) and (f_x[1] > x_2_s_median):
            if q_4_count > quadrant_mean:
                add_bool = False
        if (f_x[0] > x_1_s_median) and (f_x[1] < x_2_s_median):
            if q_2_count > quadrant_mean:
                add_bool = False
        if (f_x[0] < x_1_s_median) and (f_x[1] > x_2_s_median):
            if q_3_count > quadrant_mean:
                add_bool = False

    return add_bool


def kernel_diversity_preservation(f_x: object, archive: object, sigma: object, density_threshold: object) -> object:
    """
    Kernel method designs neighborhood of solutions according to kernel function. N number of neighboring solutions are
    generated. Distance is calculated between solution i and population j. Fitness sharing is implemented once the
    density threshold is reached. Method promotes diversity in archive of candidate solution for pareto front
    approximation by discounting candidate's fitness value.
    :param f_x: candidate value in the objective space
    :param archive: archive of solution for possible pareto front approximation
    :param sigma: niche size
    :param density_threshold: a priori minimum threshold to be reached otherwise 0 density is used resulting in -inf:inf
    fitness values for candidate solutions
    :return:
    """
    density = 0
    for i in archive:
        density += np.linalg.norm([f_x, i])  # calculate euclidean norm of candidate solution \        # to all solutions in archive

    if density < density_threshold:
        return f_x
    else:
        m_x = []
        for items in archive:
            sh = 0
            distance = np.linalg.norm([f_x, items])
            if distance < sigma:
                sh = 1 - (distance / sigma)
                m_x.append(sh)
            else:
                m_x.append(sh)

        niche_count = np.array(m_x).sum()
        f_prime = f_x / niche_count # calculates new fitness value of candidate solution
        return f_prime.tolist()


def nearest_neighbor_diversity_preservation(f_x_: object, archive: object, crowding_threshold: object) -> object:
    """
    Nearest neighbor diversity preservation method is employed using the manhattan norm to calculate the circumference
    around any single solution in the archive. Temp archive is implemented in this solution to add candidate solution
    into archive before crowding distance (circumference around candidate solution) is calculated. Solution with the
    lowest crowding distance is selected to be removed from archive.
    :param f_x_: candidate solution
    :param archive: archive of solution for possible pareto front approximation
    :param crowding_threshold: minimum crowding distance before any candidate can be nominated for removal.
    :return:
    """
    if len(archive) < 3:
        return f_x_
    else:
        circumference = [np.inf]
        temp_arch = archive + [f_x_]
        for x in range(0, len(temp_arch) - 2):
            e_1 = np.array(temp_arch[x])
            e_2 = np.array(temp_arch[x + 2])
            d = np.linalg.norm(abs((e_2 - e_1)), 1) * 2
            circumference.append(d)
        circumference.append(np.inf)
        if min(circumference) < crowding_threshold:
            f_x = archive[circumference.index(min(circumference))]
            return f_x
        else:
            return f_x_


def neighbour_x(x: object, epsilon: object) -> object:
    """
    Method is used to generate a neighboring x for neighboring solution. As X is a real number with constraints,
    A random real number is selected from a normal distribution (mu=0, std=epsilon) to be added to X. Only neighbours
    that adhere to the constraints of the problem are feasible. First feasible neighbour generated is selected.
    :param x: current value of x
    :param epsilon: distance within neighbour is deemed feasible
    :return: neighboring x value
    """
    new_x = None
    check = False
    while not check:
        new_x = np.round(x + np.random.normal(0, epsilon, 1), 2)  # generate new neighbor based off current x
        new_f_1 = f_1(new_x)
        new_f_2 = f_2(new_x)
        eval_f = np.sqrt((new_f_1 - f_1(x)) ** 2 + (new_f_2 - f_2(x)) ** 2)  # calculate distance between x and x'
        if eval_f < epsilon:
            if -10 ** 5 < new_x < 10 ** 5:  # check that new x value falls within constraints
                check = True
    return new_x[0]


def dominance(x_1: object, x_prime: object) -> object:
    """
    This method is used to see if any two solutions dominate one another. Method applies the functions to get the
    new function value of each solution given to the method
    :param x_prime: neighbouring value of x
    :param x_1: Existing x_1 value from previous round
    :return: a tuple with boolean of prime solution dominates existing & boolean if existing solution dominates prime
    """
    # check for any two given points which solution is dominated.
    x_f_1 = f_1(x_1[0])
    x_f_2 = f_2(x_1[0])
    x_f_1_prime = f_1(x_prime)
    x_f_2_prime = f_2(x_prime)
    is_dominated_prime = False
    if (x_f_2_prime >= x_f_2) and (x_f_1_prime >= x_f_1):
        is_dominated_prime = True
    is_dominated_ = False
    if (x_f_2 >= x_f_2_prime) and (x_f_1 >= x_f_1_prime):
        is_dominated_ = True
    return is_dominated_prime, is_dominated_


def delta_e(archive: object, neighbourhood_solution: object) -> object:
    """
    Define Ã = A ∪ { x } ∪ { x 0 } and | Ã x | denotes the number of solutions in Ã that
    dominate x , therefore the change in energy between x 0 and x can be expressed as:

                     ∆E ( x 0 , x ) = | Ã 0 | − | Ã x |/ | Ã|

    :param archive: Existing archive
    :param neighbourhood_solution: neighbouring solution generated
    :return: value of delta e
    """
    dominance_results = {}
    tmp_archive = archive
    for i in list(range(len(tmp_archive))):
        if i != len(tmp_archive):
            results = dominance(tmp_archive[i], neighbourhood_solution)
            dominance_results[str(tmp_archive[i])] = results

    x_dominance_count = 0
    x_prime_dominance_count = 0
    for i, j in dominance_results.items():
        if j[1]:
            x_prime_dominance_count += 1
        if j[0]:
            x_dominance_count += 1
    delta_e_value = (x_dominance_count - x_prime_dominance_count) / (len(archive) + 1)
    return delta_e_value


def acceptance_probability(t: object, delta_e_value: object) -> object:
    """
    Calculates the acceptance probability using the following formula:
        min(1, exp(-delta_e/t))
    This function returns the minimum value between 1 and e to the power of negative delta e (change in energy)
    divided by the current temperature
    :param t: current temperature
    :param delta_e_value: output value of energy change (see detla_e method)
    :return: probability of acceptance of generated neighbourhood solution
    """
    return min(np.exp(-delta_e_value / t), 1)


def metropolis_acceptance_rule(acceptance_prob: object, random_value: object) -> object:
    """
    Applies the Metropolis acceptance rule.
    If the acceptance probability is lower than that of the random generated real number ]0,1[ the rule states
    that the candidate solution is rejected, if the acceptance probability is higher, we accept the candidate
    solution.
    :param acceptance_prob: probability (see accpetance_probability method)
    :param random_value: randomly generated threshold
    :return: Boolean value, True=Accept, False=Reject
    """
    if acceptance_prob >= random_value:
        return True
    else:
        return False


def plot_pareto_front(archive: object) -> object:
    """
    This method plots the approximate patero front achieved by the DBMOSA algorithm given a specific parameter
    :param archive: archive from DBMOSA algorithm
    :return: dataframe with values of both objective functions
    """
    P = np.matrix(archive)
    df = pd.DataFrame(data=P, columns=["f_1", "f_2"])
    df.plot.scatter(x="f_1", y="f_2")
    plt.xlim(0, 9)
    plt.ylim(0, 9)
    plt.title("Approximate Patero Front")
    plt.show()
    return df


def plot_decision_space(df: object) -> object:
    """
    Plots the decision space for the values of x and the associated functions
    :param df: dataframe containing objective function values
    :return: None
    """
    plt.scatter(x=np.sqrt(df.f_1).to_list(), y=df.f_1.to_list())
    plt.scatter(x=np.sqrt(df.f_1).to_list(), y=df.f_2.to_list())
    plt.title("Decision Space")
    plt.xlabel("x")
    plt.ylabel("Associate function output")
    # df['f_2'].plot.scatter()
    plt.show()
    return None


def main():
    alpha = 1.2
    beta = 0.95  # linear cooling/reheating should be 0.05 as it linearly get reduced
    i_max = np.nan  # max number of iterations
    i = 0
    c_max = 10  # max number of acceptance
    c = 0
    d_max = 5  # max number of rejections
    d = 0
    x_1 = 0
    # T = 1.0   # 0.8 < T <= 1, with neighboring generator no neighbor gets rejected
    sigma = 3.5581572246029913
    T = (-3 * sigma) / np.log2(.81)
    min_T = 0.1
    epsilon = np.sqrt(np.power(10, 5)) / 100
    archive = []
    stopping_criterion = False
    kernel_diversity_flag = False
    nearest_neighbor_flag = False
    histogram_diversity_flag = True
    while not stopping_criterion:
        if d == d_max:  # stopping criterion for max rejections
            T = geometric_reheating_schedule(alpha, T)  # if max number of rejections is reached, increase temperature
            # T = linear_reheating_schedule(alpha, T)  # if max number of rejections is reached, increase temperature
            i += 1
            c = 0
            d = 0
            stopping_criterion = True
            logger.info(archive)

        if c == c_max:
            T = geometric_cooling_schedule(beta, T)  # if max number of acceptance is reached, reduce temperature
            # T = linear_cooling_schedule(beta, T)  # if max number of acceptance is reached, reduce temperature
            i += 1
            c = 0
            d = 0

        if not np.isnan(i_max):  # if max iterations (max_i) is set, the following
            if i == i_max:  # stopping criterion for max iterations
                stopping_criterion = True
                logger.info(archive)

        if T == min_T:  # stopping criterion for minimum iterations
            i += 1
            c = 0
            d = 0
            stopping_criterion = True

        if not archive:  # if archive has not been initialise, add first objective function values to archive
            f_1_ = f_1(x_1)
            f_2_ = f_2(x_1)
            archive.append([f_1_, f_2_])

        neighboring_solution = neighbour_x(x_1, epsilon)  # generate neighbours
        metropolis_random_value = np.random.rand()  # ger metropolis acceptance threshold
        if len(archive) != 0 and not stopping_criterion:
            detla_e_value = delta_e(archive, neighboring_solution)  # calculate delta_3
            # get output from metropolis rule
            if not metropolis_acceptance_rule(acceptance_probability(T, detla_e_value), metropolis_random_value):
                logger.info("Reject")
                i += 1  # increase epoch
                d += 1  # increase number of rejections
            else:
                x_1 = neighboring_solution
                if c != c_max:
                    # Check to remove before adding to Archive
                    if kernel_diversity_flag:
                        f_x = kernel_diversity_preservation([f_1(neighboring_solution), f_2(neighboring_solution)],
                                                            archive=archive,
                                                            sigma=.75,  # sigma=niche-size impacts diversity
                                                            density_threshold=5000)
                        for a in archive:
                            if a[0] == f_x[0] or a[1] == f_x[1]:
                                archive.remove(a)
                        # add accepted candidate solution to archive
                        archive.append(f_x)
                    if nearest_neighbor_flag:
                        f_x = nearest_neighbor_diversity_preservation([f_1(neighboring_solution),
                                                                       f_2(neighboring_solution)],
                                                                      archive,
                                                                      crowding_threshold=1.5)
                        if f_x in archive:
                            archive.remove(f_x)
                        # add accepted candidate solution to archive
                        archive.append(f_x)
                    if histogram_diversity_flag:
                        f_x = histogram_diversity_preservation([f_1(neighboring_solution),
                                                                f_2(neighboring_solution)],
                                                               archive, 100)
                        # add accepted candidate solution to archive
                        if f_x:
                            archive.append([f_1(neighboring_solution), f_2(neighboring_solution)])

                    # add accepted candidate solution to archive
                c += 1  # increase number of acceptance
                i += 1  # increase epoch

    # plot approximate pareto front gained from DBMOSA
    df = plot_pareto_front(archive)
    plot_decision_space(df)
    pd.set_option('display.max_rows', 500)
    logger.info(df)
    df['f_diff'] = df.f_1 - df.f_2
    sigma_ = df['f_diff'].std()
    logger.info(sigma_)
    return None


if __name__ == '__main__':
    main()
