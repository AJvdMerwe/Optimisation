__author__ = 'Arneau Jacques van der Merwe'

import numpy as np
from logger_setup.logger import logger


def pso(s: object, x_min: object, x_max: object, y_min: object, y_max: object, inertia: object, c_1: object, c_2: object, obj_f: object) -> object:
    particles = np.array(np.zeros((s, 2)))
    particles_new = np.array(np.zeros((s, 2)))
    velocities = np.array(np.zeros((s, 2)))
    PBest = np.array(np.zeros((s, 3)))
    GBest = np.array([0, 0, 0])
    # GBestPast = 0
    # delta = 0
    # delta1 = 0
    # delta2 = 0

    # Initialize Swarm
    min_ = 1000000
    max_no_change = 0

    for i in list(range(s)):
        particles[i, 0] = np.random.uniform(x_min, x_max, 1)
        PBest[i, 0] = particles[i, 0]
        particles[i, 1] = np.random.uniform(y_min, y_max, 1)
        PBest[i, 1] = particles[i, 1]
        PBest[i, 2] = eval(obj_f.format(particles[i, 0], particles[i, 1]))
        if PBest[i, 2] <= min_:
            min_ = PBest[i, 2]
            GBest[0] = PBest[i, 0]
            GBest[1] = PBest[i, 1]
            GBest[2] = PBest[i, 2]
        velocities[i, 0] = np.random.uniform(-(x_max - x_min), (x_max - x_min), 1)
        velocities[i, 1] = np.random.uniform(-(y_max - y_min), (y_max - y_min), 1)

    GBestPast = GBest[2]
    # Start Particle swarm
    while max_no_change < 10:
        for i in list(range(s)):
            delta = 1
            delta1 = 1
            delta2 = 1
            for d in list(range(2)):
                rP = np.random.uniform(1, 0, 1)
                rG = np.random.uniform(1, 0, 1)
                velocities[i, d] = inertia*velocities[i, d] + c_1*rP*(PBest[i, d] - particles[i, d]) + c_2*rG*(GBest[d] - particles[i, d])
                particles_new[i, d] = particles[i, d] + velocities[i, d]

            if particles_new[i, 0] < x_min:
                delta1 = (x_min - particles[i, 0])/velocities[i, 0]
            elif particles_new[i, 0] > x_max:
                delta1 = (x_max - particles[i, 0])/velocities[i, 0]
            if particles_new[i, 1] < y_min:
                delta2 = (y_min - particles[i, 1])/velocities[i, 1]
            elif particles_new[i, 1] > y_max:
                delta2 = (y_max - particles[i, 1])/velocities[i, 1]
            delta = min(delta1, delta2)
            for d in list(range(2)):
                velocities[i, d] = velocities[i, d] * delta
                particles[i, d] = particles[i, d] + velocities[i, d]

            if eval(obj_f.format(particles[i, 0], particles[i, 1])) < PBest[i, 2]:
                PBest[i, 0] = particles[i, 0]
                PBest[i, 1] = particles[i, 1]
                PBest[i, 2] = eval(obj_f.format(particles[i, 0], particles[i, 1]))

            if PBest[i, 2] < GBest[2]:
                GBest[0] = PBest[i, 0]
                GBest[1] = PBest[i, 1]
                GBest[2] = PBest[i, 2]
            logger.info(GBest)
        if GBestPast == GBest[2]:
            max_no_change += 1
        else:
            max_no_change = 0
            GBestPast = GBest[2]
    return GBestPast


def main():
    S = 50
    C_1 = 1
    C_2 = 1
    inertia = 1
    # C_1 = 1.62421000e+00
    # C_2 = 2.59158586e+00
    # inertia = 6.05988443e-01
    objective_function = '-({1}+47)*np.sin(np.sqrt(np.abs({1}+({0}/2)+47)))-{0}*np.sin(np.sqrt(np.abs({0}-({1}+47))))'
    x_min = -512
    y_min = -512
    x_max = 512
    y_max = 512
    p = pso(s=S, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, inertia=inertia, c_1=C_1, c_2=C_2,
        obj_f=objective_function)


if __name__ == '__main__':
    main()
