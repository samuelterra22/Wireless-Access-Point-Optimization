#  https://am207.github.io/2017/wiki/lab4.html

import math
from functools import partial

import numpy as np


def sa(energyfunc, initials, epochs, tempfunc, iterfunc, proposalfunc):
    accumulator = []
    best_solution = old_solution = initials['solution']
    T = initials['T']
    length = initials['length']
    best_energy = old_energy = energyfunc(old_solution)
    accepted = 0
    total = 0
    for index in range(epochs):
        print("Epoch", index)
        if index > 0:
            T = tempfunc(T)
            length = iterfunc(length)
        print("Temperature", T, "Length", length)
        for it in range(length):
            total += 1
            new_solution = proposalfunc(old_solution)
            new_energy = energyfunc(new_solution)
            # Use a min here as you could get a "probability" > 1
            alpha = min(1, np.exp((old_energy - new_energy) / T))
            if ((new_energy < old_energy) or (np.random.uniform() < alpha)):
                # Accept proposed solution
                accepted += 1
                accumulator.append((T, new_solution, new_energy))
                if new_energy < best_energy:
                    # Replace previous best with this one
                    best_energy = new_energy
                    best_solution = new_solution
                    best_index = total
                    best_temp = T
                old_energy = new_energy
                old_solution = new_solution
            else:
                # Keep the old stuff
                accumulator.append((T, old_solution, old_energy))

    best_meta = dict(index=best_index, temp=best_temp)
    print("frac accepted", accepted / total, "total iterations", total, 'bmeta', best_meta)
    return best_meta, best_solution, best_energy, accumulator

tf = lambda t: 0.8*t #temperature function
itf = lambda length: math.ceil(1.2*length) #iteration function

pfxs = lambda s, x: x + s*np.random.normal()
pfxs(0.1, 10)

pf = partial(pfxs, 0.1)
pf(10)

f = lambda x: x**2 + 4*np.sin(2*x)

inits=dict(solution=8, length=100, T=100)
bmeta, bs, be, out = sa(f, inits, 30, tf, itf, pf)

