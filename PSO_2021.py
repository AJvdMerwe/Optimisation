__author__ = 'Arneau van der Merwe'
__Date__ = '2021-09-16'

import random
import numpy as np
from logger_setup.logger import logger


def initialize_swarm(s, p, max_values, min_values, objective_function, min_):
    """
    Method is used at the start of PSO implementation to initialize Swarm. Velocities and Particles are initialized
    with random values generated within the minimum and maximum values given for each parameter.
    :param s: Number of particles in swarm
    :param max_values: List of Max values for parameters present in objective function
    :param min_values: List of Min values for parameters present in objective function
    :param objective_function: Objective function to be optimized passed as a string
    :param p: Number of variables in objective function
    :param min_: Minimum value for initialization
    :return: Global Historic Best Achieved, Global Best solution (variiable values and objective function value,
     numpy array containing values for all particles, numpy array containing values for all velocities
    """
    # Initializing Swarm
    particles = np.array(np.zeros((s, p)))
    velocities = np.array(np.zeros((s, p)))
    PBest = np.array(np.zeros((s, p + 1)))
    GBest = np.array(np.zeros((p + 1)))
    for i in list(range(s)):  # For every particle in the swarm, generate random value within bounds
        for max_value, min_value in zip(max_values, min_values):
            # for each parameter of each particle, generate value
            particles[i, max_values.index(max_value)] = np.random.uniform(min_value, max_value, 1)
            # initialize particle best to current particle values
            PBest[i, max_values.index(max_value)] = particles[i, max_values.index(max_value)]
        # Calculate objective function value for given particle
        PBest[i, len(max_values)] = eval(objective_function.format(*particles[i]))
        if PBest[i, len(max_values)] <= min_:
            # set new minimum to current particles objective value, if value is lower than current minimum
            min_ = PBest[i, len(max_values)]
            for j in list(range(len(GBest))):
                # initialize global best based on current best found candidate
                GBest[j] = PBest[i, j]
        for p in list(range(p)):
            # initialize particle velocity given particle limits given
            velocities[i, p] = np.random.uniform(-(max_values[p] - min_values[p]), (max_values[p] - min_values[p]))
    return particles, velocities, PBest, GBest


def pso(s, max_values, min_values, inertia, c_1, c_2, obj_f, p, max_no_improvement):
    """
    This is a generic and flexible implementation of the Particle Swarm Optimizer (PSO). This implementation
    takes into account all the parameters for the PSO algorithm and scales accordingly. Method can be used for
    hybrid optimization methods.
    :param s: Number of particles in swarm
    :param max_values: List of Max values for parameters present in objective function
    :param min_values: List of Min values for parameters present in objective function
    :param inertia: Inertia term for PSO algorithm
    :param c_1: Particle stochastic acceleration weight toward Particle Best
    :param c_2: Particle stochastic acceleration weight toward Global Best
    :param obj_f: Objective function to be optimized passed as a string
    :param p: Number of variables in objective function
    :param max_no_improvement: Dynamic stopping criterion, maximum number of no improvements of Swarm
    :return: Global Historic Best Achieved, Global Best solution (variable values and objective function value,
     numpy array containing values for all particles, numpy array containing values for all velocities
    """
    particles_new = np.array(np.zeros((s, p)))
    min_ = 1000000
    max_no_change = 0
    # Initialize Swarm
    particles, velocities, PBest, GBest = initialize_swarm(s, p, max_values, min_values, obj_f, min_)
    GBestPast = GBest[p]
    # Start Particle swarm
    while max_no_change < max_no_improvement:
        for i in list(range(s)):
            for d in list(range(p)):
                rP = np.random.uniform(1, 0, 1)  # get random value for rho Particle
                rG = np.random.uniform(1, 0, 1)  # get random value for rho Global
                # Calculate velocity given particle values of previous time step (t-1)
                velocities[i, d] = inertia * velocities[i, d] + c_1 * rP * (
                        PBest[i, d] - particles[i, d]) + c_2 * rG * (GBest[d] - particles[i, d])
                # Calculate new particle position based off current position and new velocity
                particles_new[i, d] = particles[i, d] + velocities[i, d]
            deltas = np.ones(p)
            # Calculate difference between particle position and minimum value, works for n number or parameters in
            # objective function
            for d in list(range(p)):
                if particles_new[i, d] < min_values[d]:
                    deltas[d] = (min_values[d] - particles[i, d]) / velocities[i, d]
                elif particles_new[i, d] > max_values[d]:
                    deltas[d] = (max_values[d] - particles[i, d]) / velocities[i, d]
            delta = min(deltas)

            # Adjust velocities based off detlas for each particle and velocity parameter. Updates partilce_parameter
            # velocities and particle values. Works for n number or parameters in objective function
            for d in list(range(p)):
                velocities[i, d] = velocities[i, d] * delta
                particles[i, d] = particles[i, d] + velocities[i, d]
            # Evaluate if objective function value is greater than current particle best objective function value
            if eval(obj_f.format(*particles[i])) < PBest[i, p]:
                for d in list(range(p + 1)):  # updates Particle best with new values
                    if d == p:  # if the parameter value is at the last instance, recalculate objective function value
                        PBest[i, d] = eval(obj_f.format(*particles[i]))
                    else:
                        PBest[i, d] = particles[i, d]  # update particle best parameter value with current particle
                        # parameter value
            # if current particle objective function value is better than global objective function value, update
            if PBest[i, p] < GBest[p]:
                for d in list(range(p + 1)):
                    GBest[d] = PBest[i, d]
            # outputs current Global best parameters and objective function
            logger.info(GBest)
        # stopping criterion, if Global Best has not improved, increment max_no_change
        if GBestPast == GBest[p]:
            max_no_change += 1
        else:
            max_no_change = 0  # resets counter of no changes to global best objective function
            GBestPast = GBest[p]
    return GBestPast, GBest, particles, velocities


def low_level_pso(w, c1, c2):
    """
    Method is to call low level Particle Swarm Optimizer (PSO), given the missing parameters. Method is also used as the
    objective function for the high level Particle Swarm Optimizer(PSO) and returns the Global Best achieved from the
    low level PSO
    :param w: Inertia/Momentum term used by the PSO
    :param c1: Particle stochastic acceleration weight toward Particle Best
    :param c2: Particle stochastic acceleration weight toward Global Best
    :return: Global best achieved by optimized objective function
    """
    S = 50
    C_1 = c1
    C_2 = c2
    inertia = w
    objective_function = '-({1}+47)*np.sin(np.sqrt(np.abs({1}+({0}/2)+47)))-{0}*np.sin(np.sqrt(np.abs({0}-({1}+47))))'
    x_min = -512
    y_min = -512
    x_max = 512
    y_max = 512
    p, _, _, _ = pso(s=S, max_values=[x_max, y_max], min_values=[x_min, y_min], inertia=inertia, c_1=C_1, c_2=C_2,
                     obj_f=objective_function, p=2, max_no_improvement=10)

    return p


def local_search(initial_canidate, obj_func, constraints):
    """

    :param initial_canidate:
    :param obj_func:
    :return:
    """
    incumbent_fitness = eval(obj_func.format(*initial_canidate))
    logger.info("Initial candidate solution: " + str(initial_canidate) + ' ' + str(incumbent_fitness) + '\n')
    stop_search = False
    incumbent = initial_canidate
    while not stop_search:
        neighborhood = generate_neighborhood(obj_func=obj_func, size=10, incumbent=incumbent,
                                             constraints=constraints)  # generates canidates
        incumbent_fitness = eval(obj_func.format(*incumbent)) # calculates best known fitness
        if all(neighbor[2] >= incumbent_fitness for neighbor in neighborhood):  # check if neighborhood has at least
            # one element that improves incumbent
            logger.info('\n Neighborhood Generate:')
            logger.info(np.matrix(neighborhood))
            stop_search = True
        else:
            logger.info('\n Neighborhood Generate:')
            logger.info(np.matrix(neighborhood))
            logger.info()
            select_neighbor = random.choice(neighborhood)  # uses library random to choose random element from list
            # select_neighbor = best_improvement(obj_func=obj_func, neighborhood=neighborhood,
            #                                    incumbent_fitness=incumbent_fitness)
            if select_neighbor[2] > incumbent_fitness:
                stop_search = True

            else:
                logger.info("selected_neighbor " + str(select_neighbor))
                incumbent = select_neighbor[:2]
    return None


def best_improvement(obj_func, neighborhood, incumbent_fitness):
    """
    The best neighbour is selected. Hence the neighbourhood is generated exhaustively. This may be prohibitively
    time-consuming for large neighbourhoods, in which case neighbourhoods may be truncated.
    :param obj_func: objective function
    :param neighborhood: candidate neighborhood
    :param incumbent_fitness: Incumbent's fitness value
    :return: best improving neighbor
    """
    selected_neighbor = []
    temp_best = incumbent_fitness
    for i in neighborhood:
        neighbor_fitness = eval(obj_func.format(*i))  # evaluates candidates' fitness
        if neighbor_fitness < temp_best:
            temp_best < neighborhood # only select new candidate if the candidate is best known solution to problem
            selected_neighbor = i  # assign best known candidate as selected candidate
    return selected_neighbor


def generate_neighborhood(obj_func, size, incumbent, constraints):
    """
    Neighborhood of size N gets generated within a unit interval of 1 from the existing incumbent's values.
    :param obj_func: objective function
    :param size: neighborhood size
    :param incumbent: Incumbent's values
    :return: generated neighborhood
    """
    neighborhood = []
    for i in range(0, size):
        neighbor_x = generate_neighbor(incumbent[0], min_val=constraints[0], max_val=constraints[1])  # generates neighbor of value of x
        neighbor_y = generate_neighbor(incumbent[1], min_val=constraints[0], max_val=constraints[1])  # generates neighbor of value of y
        neighbor_fitness = eval(obj_func.format(*[str(neighbor_x), str(neighbor_y)]))  # calculate fitness value
        neighborhood.append([neighbor_x, neighbor_y, neighbor_fitness])  # adds valid neighboring solution to
        # neighborhood
    return neighborhood


def generate_neighbor(x, min_val, max_val):
    """
    Neighbor generation algorithm. takes a single value and generate a neighboring solution using a
    gaussian normal distribution that gets added to existing value
    :param x: variable of incumbent
    :return:
    """
    acceptable = False
    selected_candidate = None
    while not acceptable:
        candidate_neighbor = np.round(x + np.random.normal(0, 1), 2)
        if min_val <= candidate_neighbor <= max_val: # check that value is between constraints set in problem
            acceptable = True
            selected_candidate = candidate_neighbor
    return selected_candidate


def question_2():
    S = 50
    C_1 = 1.62421000e+00
    C_2 = 2.59158586e+00
    inertia = 6.05988443e-01
    objective_function = '-({1}+47)*np.sin(np.sqrt(np.abs({1}+({0}/2)+47)))-{0}*np.sin(np.sqrt(np.abs({0}-({1}+47))))'
    x_min = -512
    y_min = -512
    x_max = 512
    y_max = 512
    p, gbest_high_level, particles, _ = pso(s=S, max_values=[x_max, y_max], min_values=[x_min, y_min], inertia=inertia,
                                            c_1=C_1, c_2=C_2, obj_f=objective_function, p=2, max_no_improvement=10)
    logger.info(particles)

    import pandas as pd
    particles_df = pd.DataFrame(particles[:,:2], columns=['x', 'y'])
    sub_population = particles_df.sample(n=5)
    logger.info()
    logger.info(sub_population)
    logger.info()
    for i in sub_population.values:
        local_search(i, obj_func=objective_function, constraints=[x_min, x_max])
        logger.info()
    return None


def main():
    # Run High level PSO
    S = 10
    C_1 = 1
    C_2 = 1
    inertia = 1
    objective_function = 'low_level_pso({0}, {1}, {2})'  # Use low level PSO as objective function to optimize
    w_min = 0.4
    w_max = 0.9
    c1_min = 0
    c1_max = 2.15
    c2_min = 0
    c2_max = 2.85
    p, gbest_high_level, particles, _ = pso(s=S, max_values=[w_max, c1_max, c2_max], min_values=[w_min, c1_min, c2_min],
                                            inertia=inertia, c_1=C_1, c_2=C_2, obj_f=objective_function,
                                            p=3, max_no_improvement=10)

    # This was maually run after Hybrid PSO completed to get particles of low level PSO given optimal w*,c_1* and c_2*
    # Best High Level Parameters 6.05988443e-01  1.62421000e+00  2.59158586e+00 -9.59640663e+02
    S = 50
    C_1 = 1.62421000e+00
    C_2 = 2.59158586e+00
    inertia = 6.05988443e-01
    objective_function = '-({1}+47)*np.sin(np.sqrt(np.abs({1}+({0}/2)+47)))-{0}*np.sin(np.sqrt(np.abs({0}-({1}+47))))'
    x_min = -512
    y_min = -512
    x_max = 512
    y_max = 512
    g_best_history, gbest, particles, v = pso(s=S, max_values=[x_max, y_max], min_values=[x_min, y_min],
                                              inertia=inertia, c_1=C_1, c_2=C_2,
                                              obj_f=objective_function, p=2, max_no_improvement=10)
    logger.info(particles)
    import pandas as pd
    import matplotlib.pyplot as plt
    particles_df = pd.DataFrame(particles, columns=['x', 'y'])
    particles_df.plot.scatter(x='x', y='y')
    plt.show()
    # particles_df.to_csv('PSO_low_level.csv')

    # question_2()


if __name__ == '__main__':
    main()
