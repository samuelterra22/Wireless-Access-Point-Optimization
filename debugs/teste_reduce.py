#!/usr/bin/python
# -*- coding: latin1 -*-


matrix = np.ones(shape=(5, 5))
sum_reduce = cuda.reduce(lambda a, b: a + b)
soma_mw = sum_reduce(np.array([10**(x/10.) for line in matrix for x in line]))

print(soma_mw)