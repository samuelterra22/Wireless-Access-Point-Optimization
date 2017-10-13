import numpy
from numba import cuda


@cuda.reduce
def sum_reduce(a, b):
    return a + b


A = (numpy.arange(1234, dtype=numpy.float64)) + 1
expect = A.sum()  # numpy sum reduction
got = sum_reduce(A)  # cuda sum reduction
assert expect == got
print(expect == got)