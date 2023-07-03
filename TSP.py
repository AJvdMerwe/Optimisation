__author__ = 'Arneau Jacques van der Merwe'

import numpy as np
import itertools
import random
from logger_setup.logger import logger


def tournament_parent_selector(population: object, k: object, n: object, distance_matrix: object) -> object:
    """
    Implementing tournament selection, K candidates in population gets selected. This method runs the tournament
    selection multiple time to produce N number of tournament winners. Uses python random.sampling method.
    :param distance_matrix: matrix with cost of between cities in TSP
    :param n: number of winners that gets finally selected
    :param population: list of candidate solution population
    :param k: Number to be randomly selected
    :return: randomly selected sub-population
    """
    tournament_winners = []
    for i in list(range(n)):
        population_fitness_dict, selected_group = tournament_selection(population, k, distance_matrix)
        sort_orders = sorted(population_fitness_dict.items(), key=lambda x: x[1], reverse=False)
        tournament_winners.append(selected_group[sort_orders[0][0]])
        population.remove(selected_group[sort_orders[0][0]])
    return tournament_winners


def tournament_selection(population: object, k: object, distance_matrix: object) -> object:
    """
    Using tournament selection, K candidates in population gets selected. Uses python random.sampling method.
    :param population: list of candidate solution population
    :param k: Number to be randomly selected
    :return: randomly selected sub-population
    :return: fitness of subpopulation
    """
    population_fitness_dict = {}
    selected_group = random.sample(population, k)
    fitness_values = []
    for p, i in zip(selected_group, list(range(len(population)))):
        fitness_values.append(calc_fitness(distance_matrix, p))
        population_fitness_dict[i] = calc_fitness(distance_matrix, p)
    return population_fitness_dict, selected_group


def calc_fitness(distance_matrix: object, candidate_solution: object) -> object:
    """
    Method calculates the distance of traveled in the Traveling Salesman Problem, given that a candidate solution
    is provided, as well as the distance matrix. Candidate solution is some permutation of all cities in the problem.
    When the last city is reached, the distance between the last city and the starting city is added.
    :param distance_matrix: Distance Matrix in numpy.matrix form
    :param candidate_solution: list of candidate solution
    :return: distance traveled for a given candidate solution
    """
    total_distance = 0
    for i in range(len(candidate_solution)):
        if i == len(candidate_solution):
            next_city = candidate_solution[0]
        else:
            next_city = candidate_solution[i]
        current_city = candidate_solution[i-1]
        total_distance += distance_matrix[(next_city-1), (current_city-1)]
    return total_distance


def cross_over_operator(opt: object, parents: object) -> object:
    """
    K-opt cross-over operator is employed. K is the number of cuts to be made to each parent. The k-opt takes the
    elements from the first parents from the start to the first cut point, and from the second cut-point to the end
    of the candidate solution. The selected elements are removed from the second parent and the remainder is added to
    those selected from parent 1 to create the first offspring. The same process is done for offspring 2, replacing
    parent 1 with parent 2.

    This method also applies the mutation method to a randomly selected offspring.
    :param opt: number of cuts to be made
    :param parents: list of 2 candidate solutions
    :return: list of offspring candidate solutions
    """
    cuts_ = list(range(0, len(parents[0])))
    cuts = sorted(random.sample(cuts_, opt))
    fill_1 = [i for i in parents[1] if i not in parents[0][:cuts[0]] + parents[0][cuts[1]:]]
    fill_2 = [i for i in parents[0] if i not in parents[1][:cuts[0]] + parents[1][cuts[1]:]]
    off_spring_1 = parents[0][:cuts[0]] + fill_1 + parents[0][cuts[1]:]
    off_spring_2 = parents[1][:cuts[0]] + fill_2 + parents[1][cuts[1]:]
    off_spring = [off_spring_1, off_spring_2]

    return off_spring


def mutation(offspring: object) -> object:
    """
    Mutation takes 2 randomly selected elements in a candidate solution and swaps these to elements in place to create
    a new mutated offspring
    :param offspring: list, candidate solution
    :return: mutated candidate solution
    """
    positions = list(range(len(offspring)))
    cuts = sorted(random.sample(positions, 2))
    offspring[cuts[0]], offspring[cuts[1]] = offspring[cuts[1]], offspring[cuts[0]]
    return offspring


def elitism(population: object, distance_matrix: object, n: object) -> object:
    """
    An Elitism selection strategy is employed. The top performing n number of population members are selected to
    form part of the new population. The fitness value of the population members are calculated using the calc_fitness
    method and ranked ascending for minimization problems like TSP. The fitness values, along with winning candidate
    solutions are returned
    :param n: Number of winning candidate solution to move to next generation
    :param population: list of list, containing candidate solutions, parents and offspring
    :param distance_matrix: distance matrix that is of class numpy.matrix()
    :return: list containing list of tuple of winning fitness values, as well as winning candidate solutions
    """
    fitness_values = []
    population_fitness_dict = {}
    for p, i in zip(population, list(range(len(population)))):
        fitness_values.append(calc_fitness(distance_matrix, p))
        population_fitness_dict[i] = calc_fitness(distance_matrix, p)
    sort_orders = sorted(population_fitness_dict.items(), key=lambda x: x[1], reverse=False)
    best_candidate_solutions = sort_orders[:n]
    winning_populations = []
    for i in list(range(8)):
        winning_populations.append(population[sort_orders[i][0]])
    return best_candidate_solutions, winning_populations


def main():
    # initialize distance matrix for TSP
    distance_matrix = np.matrix([[0, 41, 26, 31, 27, 35],
                                 [41, 0, 29, 32, 40, 33],
                                 [26, 29, 0, 25, 34, 42],
                                 [31, 32, 25, 0, 28, 34],
                                 [27, 40, 34, 28, 0, 36],
                                 [35, 33, 42, 34, 36, 0]])

    city_number = [1, 2, 3, 4, 5, 6]
    permutations_ = list(itertools.permutations(city_number, 6))  # get all possible permutations for problem
    population = random.sample(permutations_, 8)  # select initial population
    logger.info("\nGenerated Population")
    logger.info(population)
    logger.info("\n Fitness values for Population")
    for i in population:
        logger.info(calc_fitness(distance_matrix, i))
    # implement tournament selection and select 3 candidates
    generation_counter = 0
    z_hat = 0
    delta_z = 0
    new_population = population
    while generation_counter <= 10:
        off_spring = []
        # new_population = []
        logger.info("\nSelected Parents:")
        tournament_winner = tournament_parent_selector(new_population, k=3, n=6, distance_matrix=distance_matrix)
        logger.info(tournament_winner)
        new_population = []
        # for i in list(itertools.permutations(tournament_winner, 2)):  # create permutation with 3 candidates
        for i in list(range(0, len(tournament_winner), 2)):
            new_population.append(list(tournament_winner[i]))
            new_population.append(list(tournament_winner[i+1]))
            ofs_1, ofs_2 = cross_over_operator(opt=2,
                                               parents=[list(tournament_winner[i]),
                                                list(tournament_winner[i+1])])
            # apply crossover
            off_spring.append(ofs_1)
            off_spring.append(ofs_2)
        logger.info("\nOffspring generater:")
        logger.info(off_spring)
        random_selection = random.sample(range(len(off_spring)), 1)
        logger.info("\nOffspring selected for mutation:")
        logger.info(off_spring[random_selection[0]])
        off_spring[random_selection[0]] = mutation(off_spring[random_selection[0]])
        logger.info("\nMutated Offspring:")
        logger.info(off_spring[random_selection[0]])
        new_population = new_population + off_spring
        _, new_population = elitism(new_population, distance_matrix, 8)  # select best performing member of population
        logger.info("\nNew seclected population")
        logger.info(new_population)
        logger.info("\nAssociated Fitness values:")
        logger.info(_)
        top_fitness_values = []
        for i in _:
            top_fitness_values.append(i[1])
        logger.info("\nMean Fitness value for surviving population:")
        logger.info(np.mean(np.array(top_fitness_values)))
        new_best_solution = list(_[0])[1]  # get incumbent from current population
        if z_hat != 0:
            if z_hat > new_best_solution:  # if new population best is greater than incumbent, it becomes incumbent
                z_hat = new_best_solution
            delta_z = z_hat - new_best_solution  # change in incumbent is calculated
        else:
            z_hat = new_best_solution  # if new population best is greater than incumbent, it becomes incumbent
        if delta_z == 0:  # generation counter is increase if detla_z = 0
            generation_counter += 1
            # if generation_counter is increased, no change was made to incumbent. if this occurs 10 times, optimisation
            # search will end.


if __name__ == '__main__':
    main()
