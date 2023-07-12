# Optimisation
Optimisation algorithms

This directory contains coded solutions to various meta-heuristic optimisation problems.

# Knapsack problem 
Implement, in a computer language of your choice, the method of local search for solving the
following instance of the knapsack problem.

    Maximise z = 8s1 + 12s2 + 9s3 + 14s4 + 16s5 + 10s6 + 6s7 + 7s8 + 11s9 + 13s10
    such that 3s1 + 2s2 + s3 + 4s4 + 3s5 + 3s6 + s7 + 2s8 + 2s9 + 5s10 ≤ 12,
                s1 , s2 , s3 , s4 , s5 , s6 , s7 , s8 , s9 , s10 ∈ {0, 1}.

Encode candidate solutions as binary vectors of the form [s1 , s2 , s3 , s4 , s5 , s6 , s7 , s8 , s9 , s10 ],
employ single-bit complement moves and use the method of best improvement for neighbour
selection

# DBMOSA

Implement, in a computer language of your choice, the method of dominance-based multi-objective simulated
annealing (DBMOSA) for finding an approximation of the following continuous multi-objective optimisation
problem’s Pareto front.

        Minimisef1 (x) = x2 ,
        minimisef2 (x) = (x − 2)2 ,
        subject to− 105 ≤ x ≤ 105 .
        
For the method of simulated annealing determine its appropriate parameter values and operator settings
empirically, they are:
  • Starting temperature T0 , e.g. accept all, acceptance deviation, or acceptance ratio.
  • Epoch length, e.g. static or dynamic paradigm.
  • Cooling and reheating schedule, e.g. linear, geometric, logarithmic, or adaptive, to name a few.
  • Search termination criterion, e.g. reaching a final temperature, reaching a pre-determined number of
iterations without improvement, or achieving a small number of move acceptances over a pre-specified
successive number of epochs.

# Particle Swarm Optimisation and Hybrid meta-heuristic optimisation

Starting with the low-level, the low-level PSO is tasked to optimise the following objective
function.
r
q
x
Minimise f ( x, y) = −(y + 47) sin ( |y + + 47|) − x sin ( | x − (y + 47)|)
2
Subject to − 512 ≥ x, y, ≥ 512
With the information given about the objective function, all values except w, c1, c2 are avail-
able to pass to the low-level PSO. These values will be given by the high-level PSO in the
meta-optimisation algorithm. We also know that for the given function, the global minima is

                  f (512, 404.2319) = 959.6407
                  
The high-level PSO implementation will use the low-level PSO as the objective function to
minimize. The low-level PSO will return it’s global best objective function value as the objec-
tive function to the high-level PSO. As w, c1, c2 are needed for the low-level PSO, the high-level
PSO will need to solve the minimization problem in 3 dimensions.

Each particle will have 3 values, namely w, c1, c2, which will be passed to the low-level PSO.
The values of w, c1, c2 for the high-level PSO are given as w = 1, c1 = 1, c2 = 1. The limits for
the variables of the objective function is not given. There has been extensive research done on
the initial parameter values for the PSO algorithm as they impact the algorithms performance.
The following limits where chosen:

                          0.4 ≥ w ≥ 0.9
                          0 ≥ c1 ≥ 2.15
                          0 ≥ c2 ≥ 2.85
                          
(He et al., 2016)
