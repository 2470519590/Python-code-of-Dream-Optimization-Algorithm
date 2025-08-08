import numpy as np
import DOA
import matplotlib.pyplot as plt


def func(x):
    return np.sum(x**0.5)


dim = 100
pop = 20
maxIter = 1000
lb = np.array([0]*dim)
ub = np.array([5]*dim)
fobj = func
u = 0.9

fbest, sbest, fbest_history = DOA.DOA(pop, maxIter, lb, ub, dim, func_obj=fobj, u=u)

print("fbest:", fbest)
print("sbest:", sbest)
#print(fbest_history)
plt.plot(fbest_history[0, :])

plt.xlabel('Iteration')
plt.ylabel('FuncValueBest')
plt.show()