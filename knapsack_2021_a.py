__author__ = 'Arneau Jacques van der Merwe'

import numpy as np
from logger_setup.logger import logger


def bit_complement(objective_function_values: object) -> object:
    """
    This function applies the single bit-complement flip. This function takes either the initial_s_0 or
    takes the best candidate solution from the previous iteration (T). It then produces the new neighbourhood
    of candidate solution by flipping a single bit in the solution.
    :param objective_function_values: list of binary integer values
    :return: list of lists, candidate solution based off previous best solution
    """
    candidate_solutions = []
    for i in list(range(len(objective_function_values))):
        tmp_solution = objective_function_values.copy()
        if tmp_solution[i] == 0:
            tmp_solution[i] = 1
        else:
            tmp_solution[i] = 0
        candidate_solutions.append(tmp_solution)
    return candidate_solutions


def calculate_weights(objective_function_weights: object, candidate_solutions: object) -> object:
    """
    This function takes the specified weights of a knapsack problem that is used to calculate the weight for a
    give solution. It takes the candidate solution and multiplies the candidate solution with the weights to
    calculate and output the new z value for a given solution.
    :param objective_function_weights: list of integer/float weight for given problem
    :param candidate_solutions: list of binary integer values
    :return: calculates the new z value of the best solution
    """
    outputs = np.sum(np.array(objective_function_weights) * np.array(candidate_solutions))
    return outputs


def check_constraints(candidate_solutions: object, objectiv_function_constraint: object, constraint: object, weights: object) -> object:
    """
    Function takes list of candidate solutions, objective function constraints and total constraints. First the
    limits of any give candidate solution is calculated. This is done by multiplying the candidate solution with
    the objective function constraint. This then gets checked against the given constraint. If the candidate solution
    exceeds the constraint, the value is changed to nan. The feasible solution population is then calculated where the
    candidates that exceed the constraint is replaced with an empty solution. Once the feasible solution population is
    created, the fitness of the solutions are calculated.
    :param candidate_solutions: list of list of binary integer values, represents neighbourhood of candidate solutions
    :param objectiv_function_constraint: list of associated binary cost for knapsack problem
    :param constraint: constraint placed on knapsack problem
    :param weights: weight associated with maximization function of optimization problem
    :return: fitness of feasible solution population
    """
    limits = np.sum(np.array(candidate_solutions) * np.array(objectiv_function_constraint), axis=1)
    feasible_candidates = [i if i <= constraint else np.nan for i in limits.tolist()]
    constraint_population = [[0, 0, 0, 0, 0, 0, 0, 0, 0] if np.isnan(feasible_candidates[i])
                             else candidate_solutions[i]
                             for i in range(len(feasible_candidates))]
    feasible_population_ = np.sum(np.array(constraint_population) * weights, axis=1).tolist()
    return feasible_population_, constraint_population


def select_best_solution(feasibility: object, proposed_solutions: object) -> object:
    """
    Function selects the feasible candidate solution from population that maximises the objective function
    :param feasibility: list of limits calculated and feasibility of neighbourhood solutions
    :param proposed_solutions: list of list of binary values
    :return: candidate solution that provides the best objective function fitness value
    """
    selected_candidate = proposed_solutions[feasibility.index(max(feasibility))]
    return selected_candidate


def main():
    # objective_function_weight = [8, 12, 9, 14, 16, 10, 6, 7, 11, 13]
    # objective_function_constraints = [3, 2, 1, 4, 3, 3, 1, 2, 2, 5]
    # initial_s_0 = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    # weight_constraint = 12
    # z_hat = 0
    # local_search = True

    objective_function_weight = [8, 11, 9, 12, 14, 10, 6, 7, 13]
    objective_function_constraints = [1, 2, 3, 2, 3, 4, 1, 5, 3]
    initial_s_0 = [0, 1, 1, 1, 0, 0, 1, 1, 1]
    # initial_s_0 = [0, 1, 0, 1, 0, 0, 0, 1, 0]
    weight_constraint = 16
    z_hat = 0
    local_search = True

    candidate_solutions = []
    candidate_obj_values = []

    n = 1
    m = 100
    d = 1

    while local_search:
        # start local search, if z_hat has not been initialised, given initial_s_0 is used for s_hat
        if z_hat == 0:
            s_hat = initial_s_0
            z_hat = np.sum(np.array(initial_s_0) * np.array(objective_function_weight))
        else:
            # if z_hat has been set, best solution from previous T is taken as s_hat for this iteration
            s_hat = best_solution
        proposed_solutions = bit_complement(s_hat)  # neighbourhood is generated off s_hat
        feasible_obj_func, feasible_population = check_constraints(proposed_solutions,
                                                                   objective_function_constraints,
                                                                   weight_constraint,
                                                                   objective_function_weight)

        # only select feasible neighboring solutions
        feasible_population = [i for i in feasible_population if feasible_obj_func[feasible_population.index(i)] != 0]

        # selecting random candidate in feasible neighboring solutions regardless of quality
        random_candidate_selection = int(np.random.random(1)*len(feasible_population))
        best_solution = feasible_population[random_candidate_selection]

        best_z = calculate_weights(objective_function_weight,
                                   best_solution)  # randomly selected candidate solution fitness value is calculated
        # if best_z <= z_hat:
        candidate_obj_values.append(best_z)
        candidate_solutions.append(best_solution)
        if n == m:
            # if the new best fitness value is the same as previous fitness value, terminate local search
            local_search = False
        # if best_z >= z_hat:
        #     z_hat = best_z
        else:
            z_hat = best_z
        logger.info()
        logger.info(n)
        logger.info("best solution: " + str(best_solution))
        logger.info("best objective function value :" + str(best_z))
        n += 1

    logger.info()
    # print("proposed solutions: \n" + str(np.matrix(proposed_solutions)))
    logger.info("best feasible_population: " + str(np.matrix(candidate_solutions)))
    logger.info("best solution: " + str(best_solution))
    logger.info("best objective function value :" + str(z_hat))

    f_bar = np.array(candidate_obj_values).mean()
    sigma_f_sqr = np.array(candidate_obj_values).std()**2

    product_f_s = []
    for i in range(100 - d):
        temp = (candidate_obj_values[i] - f_bar)*(candidate_obj_values[i+d] - f_bar)
        product_f_s.append(temp)

    random_walk_corr = (1/((m - d) * sigma_f_sqr)) * np.sum(np.array(product_f_s))
    l = 1/abs((np.log(abs(random_walk_corr))))

    from scipy.spatial.distance import hamming
    hamming_distances = hamming(candidate_solutions[candidate_obj_values.index(np.array(candidate_obj_values).max())],
                                candidate_solutions[candidate_obj_values.index(np.array(candidate_obj_values).min())]) \
                        * len(candidate_obj_values)

    # from scipy.spatial.distance import hamming
    # hamming_distances = []
    # for items in candidate_solutions:
    #     hamming_distances.append(hamming(initial_s_0, items) * len(items))
    diam_g = np.array(hamming_distances)
    logger.info()
    # print(product_f_s)
    logger.info("random walk correlation function value r(1): " + str(random_walk_corr))
    logger.info("correlation length: " + str(l))
    logger.info("Diam(P): " + str(hamming_distances))
    logger.info("normalized correlation: " + str(l/diam_g))
    logger.info()


if __name__ == '__main__':
    main()
