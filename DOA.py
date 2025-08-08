import numpy as np


def randperm(Dim, k):
    return np.random.choice(np.arange(1, Dim+1), size=k, replace=False)


def initialization(pop, dim, ub, lb):
    Boundary_no = ub.size

    if dim != Boundary_no:
        print("Warning: Dimension and Boundary Mismatch")

    if Boundary_no == 1:
        initial_mat = np.random.uniform(lb, ub, (pop, dim))
        return initial_mat

    if Boundary_no > 1:
        initial_mat = np.ones((pop, Boundary_no))
        for i in range(0, dim):

            initial_mat[:, i] = np.random.uniform(lb[:, i], ub[:, i], pop)
        return initial_mat

    return None


def default_fobj(x):
    if not isinstance(x, np.ndarray) or x.ndim != 1:
        raise ValueError("Input must be one-dimensional")
    return np.sum(x ** 2)


def DOA(populations, Tmax, lower_boundary, upper_boundary, dimension, func_obj=default_fobj, u=0.9):

    # Eq.2 Eq.3 Generate the initial population
    lower_boundary = np.zeros((1, dimension)) + lower_boundary
    upper_boundary = np.zeros((1, dimension)) + upper_boundary
    X = initialization(populations, dimension, upper_boundary, lower_boundary)

    SELECT = np.array(range(1, populations + 1))

    fitness = np.ones((populations, dimension))
    vbest = np.ones((1, dimension))
    vbest_5 = np.ones((5, dimension))
    fbest = float('inf')
    fbestd = np.ones((5, 1))
    fbest_history = np.ones((1, Tmax))

    for i in range(0, 5):
        fbestd[i,] = fbest

    # exploration phase
    for t in range(1, int(u*Tmax) + 1):

        for q in range(1, 5):
            # Eq.10 calculate the number of forgetting dimensions
            kq = np.random.randint(np.ceil(dimension/8/q).astype(int), max(2, np.ceil(dimension/3/q).astype(int))+1)

            # update the best solution
            for j in range(int((q-1)/5*populations)+1, int(q/5*populations)+1):
                if func_obj(X[j-1, :]) < fbestd[q-1, :]:
                    vbest_5[q-1, :] = X[j-1, :]
                    fbestd[q-1, :] = func_obj(X[j-1, :])

            for j in range(int((q-1)/5*populations)+1,  int(q/5*populations)+1):

                # Eq.4 update X
                X[j-1, :] = vbest_5[q-1, :]

                # generate the forgetting dimensions
                In = randperm(dimension, kq)

                # Parameter u is used to adjust the ratio between forgetting supplementation and dream sharing during the exploration phase.
                # rand<u, execute the forget supplement strategy
                if np.random.rand() < u:
                    for h in range(1, kq+1):
                        # Eq.5 update X
                        X[j-1, In[h-1]-1] = (X[j-1, In[h-1]-1] +
                                             (np.random.rand() * (upper_boundary[0, In[h-1]-1] - lower_boundary[0, In[h - 1] - 1]) + lower_boundary[0, In[h - 1] - 1]) *
                                             (np.cos((t+Tmax-u*Tmax) * np.pi/Tmax) + 1) / 2)

                        # check the boundary
                        if (X[j-1, In[h-1]-1] > upper_boundary[0, In[h-1]-1]) or (X[j-1, In[h-1]-1] < lower_boundary[0, In[h - 1] - 1]):
                            # The original text explains. When dim>15, the problem becomes more dimensional and complex,
                            # with more local optimal solutions,
                            # requiring enhanced global optimization capabilities and the ability to escape from local optimal solutions.
                            # Therefore, a method similar to the dream sharing strategy used in the development phase is adopted for re updating.
                            if dimension > 15:
                                select = np.copy(SELECT)
                                np.delete(select, j-1)

                                # Eq.13 replace X[i,j]
                                m = select[np.random.randint(1, populations)-1]
                                X[j-1, In[h-1]-1] = X[m-1, In[h-1]-1]
                            else:
                                # dim<=15 There are relatively few local optima,
                                # and traditional random methods are used to update points beyond the search boundary
                                # Eq. 12
                                X[j-1, In[h-1]-1] = np.random.rand() * (upper_boundary[0, In[h-1]-1] - lower_boundary[0, In[h - 1] - 1]) + lower_boundary[0, In[h - 1] - 1]
                else:
                    # rand>=u, execute the dream sharing strategy
                    for h in range(1, kq+1):
                        # Eq.6 update X
                        X[j-1, In[h-1]-1] = X[np.random.randint(1, populations+1)-1, In[h-1]-1]

                if fbestd[q-1] < fbest:
                    fbest = fbestd[q-1]
                    vbest = vbest_5[q-1, :]

        fbest_history[0, t-1] = fbest

    # exploitation phase
    for t in range(int(u*Tmax) + 1, Tmax+1):

        for p in range(1, populations+1):
            if func_obj(X[p-1, :]) < fbest:
                vbest = X[p-1, :]
                fbest = func_obj(X[p-1, :])

        for j in range(1, populations+1):
            fitness[j-1, :] = func_obj(X[j-1, :])

            # Eq.11 update kr
            kr = np.random.randint(2, max(2, np.ceil(dimension/3).astype(int))+1)

            # Eq.7 update X
            X[j-1, :] = vbest

            In = randperm(dimension, kr)

            for h in range(1, kr+1):
                # Eq.8 update X
                X[j-1, In[h-1]-1] = (X[j-1, In[h-1]-1] +
                                     (np.random.rand() *
                                      (upper_boundary[0, In[h-1]-1] - lower_boundary[0, In[h - 1] - 1]) + lower_boundary[0, In[h - 1] - 1]) *
                                     (np.cos(t*np.pi/Tmax)+1) / 2)

                # check the boundary
                if (X[j-1, In[h-1]-1] > upper_boundary[0, In[h-1]-1]) or (X[j-1, In[h-1]-1] < lower_boundary[0, In[h - 1] - 1]):
                    if dimension > 15:
                        select = np.copy(SELECT)
                        np.delete(select, j-1)
                        m = select[np.random.randint(1, populations)-1]
                        X[j-1, In[h-1]-1] = X[m-1, In[h-1]-1]
                    else:
                        X[j-1, In[h-1]-1] = np.random.rand() * (upper_boundary[0, In[h-1]-1] - lower_boundary[0, In[h - 1] - 1]) + lower_boundary[0, In[h - 1] - 1]
        fbest_history[0, t-1] = fbest
    return fbest, vbest, fbest_history