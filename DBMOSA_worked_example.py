__author__ = 'Arneau Jacques van der Merwe'

# TODO: remove rounding before submitting or trying different version of problem

import numpy as np
# import random


def geometric_cooling_schedule(beta, t):
    return beta * t


def geometric_reheating_schedule(alpha, t):
    return alpha * t


def f_1(x):
    # TODO: replace function 1
    return x


def f_2(x_1, x_2):
    # TODO: replace function 2
    x_ = (1 + x_2) / x_1
    return x_


def neighbourhood_x_1(x_1, r_1):
    # TODO: replace neighbourhood function 1
    x_1_prime = x_1 + (0.5 - r_1) * min(0.05, 1 - x_1, x_1 - .1)
    return x_1_prime


def neighbourhood_x_2(x_2, r_2):
    # TODO: replace neighbourhood function 2
    x_2_primte = x_2 + (.5 - r_2) * min(0.2, 5 - r_2, x_2)
    return x_2_primte


def generate_neighboring_solution(x_1, x_2, r_1, r_2):
    x_1_prime = neighbourhood_x_1(x_1, r_1)
    x_2_prime = neighbourhood_x_2(x_2, r_2)
    # TODO: Remove rounding
    return np.round(x_1_prime, 2), np.round(x_2_prime, 2)


def dominance(x_1, x_2, x_1_prime, x_2_prime):
    """
    This method is used to see if any two solutions dominate one another. Method applies the functions to get the
    new function value of each solution given to the method
    :param x_1: Existing x_1 value from previous round
    :param x_2: Existing x_2 value from previous round
    :param x_1_prime: x_1_prime is the neighbouring solution generated from x_1
    :param x_2_prime: x_2_prime is the neighbouring solution generated from x_2
    :return: a tuple with boolean of prime solution dominates existing & boolean if existing solution dominates prime
    """
    # check for any two given points which solution is dominated.
    # TODO: Remove rounding
    x_1_f_1 = np.round(f_1(x_1), 2)
    x_2_f_2 = np.round(f_2(x_1, x_2), 2)
    # TODO: Remove rounding
    x_1_f_1_prime = np.round(f_1(x_1_prime), 2)
    x_2_f_2_prime = np.round(f_2(x_1_prime, x_2_prime), 2)
    is_dominated_prime = False
    if (x_2_f_2_prime >= x_2_f_2) and (x_1_f_1_prime >= x_1_f_1):
        is_dominated_prime = True
    is_dominated_ = False
    if (x_2_f_2 >= x_2_f_2_prime) and (x_1_f_1 >= x_1_f_1_prime):
        is_dominated_ = True
    return is_dominated_prime, is_dominated_


def delta_e(archive, neighbourhood_solution):
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
            results = dominance(tmp_archive[i][0], tmp_archive[i][1], neighbourhood_solution[0],
                                neighbourhood_solution[1])
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


def acceptance_probability(t, delta_e_value):
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


def metropolis_acceptance_rule(acceptance_prob, random_value):
    """
    Applies the Metropolis acceptance rule.
    If the acceptance probability is lower than that of the random generated real number ]0,1[ the rule states
    that the candidate solution is rejected, if the accpetance probability is higher, we accept the candidate
    solution.
    :param acceptance_prob: probability (see accpetance_probability method)
    :param random_value: randomly generated threshold
    :return: Boolean value, True=Accept, False=Reject
    """
    if acceptance_prob >= random_value:
        return True
    else:
        return False


def main():
    # Temprature = 1  # accept all, acceptance deviation
    # epoch = 0  # static & dynamic
    # cooling_schedule = None  # linear, geometric
    # heating_schedule = None  # linear, geometric
    # # Search_termination_criterion
    # max_epoch = 3
    # max_temp = 0.7
    # archive = [[0.50, 8.00], [0.48, 8.44], [0.46, 8.93], [0.48, 8.65]]
    # delta_e(archive)
    alpha = 1.2
    beta = 0.9
    i_max = 3
    i = 1
    c_max = 3
    c = 0
    d_max = 5
    d = 0
    x_1 = 0.5
    x_2 = 3.00
    T = 1.00
    t = 0
    t_max = 6
    archive = [[0.5, 3.00]]
    random_numbers = [0.94, 0.26, 0.98, 0.19, 0.03, 0.31, 0.84, 0.42, 0.64,
                      0.07, 0.51, 0.69, 0.23, 0.36, 0.28, 0.43, 0.67]
    r_1_values = [.94, .98, .03, .42, .07, .23, .28]
    r_2_values = [.26, .19, .31, .64, .51, .36, .43]
    stopping_criterion = False
    while not stopping_criterion:
        if d == d_max:
            T = geometric_reheating_schedule(alpha, T)  # if max number of rejections is reached, increase temperature
            i += 1
            c = 0
            d = 0
            stopping_criterion = True
            print(archive)
            break
        if c == c_max:
            T = geometric_cooling_schedule(beta, T)  # if max number of acceptance is reached, reduce temperature
            i += 1
            c = 0
            d = 0
        if t == t_max:
            print(archive)
            break
        r_1, r_2 = r_1_values[t], r_2_values[t]  # get random values for neighbour generation
        neighboring_solution = generate_neighboring_solution(x_1, x_2, r_1, r_2)  # generate neighbours
        metropolis_random_value = random_numbers[random_numbers.index(r_2) + 1]  # ger metropolis acceptance threshold
        if len(archive) != 0:
            detla_e_value = delta_e(archive, neighboring_solution)  # calculate delta_3
            # get output from metropolis rule
            if not metropolis_acceptance_rule(acceptance_probability(T, detla_e_value), metropolis_random_value):
                print("Reject")
                t += 1  # increase epoch
                d += 1  # increase number of rejections
            else:
                x_1, x_2 = neighboring_solution
                if c != c_max:
                    # Check to remove before adding to Archive
                    for a in archive:
                        if a[0] == neighboring_solution[0] or a[1] == neighboring_solution[1]:
                            archive.remove(a)
                    # add accepted candidate solution to archive
                    archive.append(list(neighboring_solution))
                c += 1  # increase number of acceptance
                t += 1  # increase epoch

    return None


if __name__ == '__main__':
    main()
