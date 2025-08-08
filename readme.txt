

Dream Optimization Algorithm (DOA) is a novel metaheuristic algorithm inspired by human dreams.



Parameters of the algorithm(in main.py):

dim:
the dimension of the problem

pop:
the population size (PS: Since the DOA algorithm internally divides the population into 5 groups,
pop requires a positive integer multiple of 5)

maxIter:
the maximum number of iterations

lb:
the lower bound of the problem (required to be a row vector with a length of dim)

ub:
the upper bound of the problem (as above)

Fobj:
the optimization objective function

u:
the proportion of the entire process in the exploration stage.
The original text states that after a large number of numerical experiments,
considering the practicality and stability of the algorithm, u is set to 0.9



About the details, please refer to the following paper:

Lang Y, Gao Y. Dream Optimization Algorithm (DOA): A novel metaheuristic optimization algorithm inspired by human dreams and its applications to real-world engineering problems[J]. Computer Methods in Applied Mechanics and Engineering, 2025, 436: 117718.

link:
https://www.sciencedirect.com/science/article/abs/pii/S0045782524009745