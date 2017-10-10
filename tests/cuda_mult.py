import numpy as np
import math
from timeit import default_timer as timer
from numba import cuda
from numba import *


def mult(a, b):
    return a * b


mult_gpu = cuda.jit(restype=float32, argtypes=[float32, float32], device=True)(mult)


@cuda.jit(argtypes=[float32[:, :], float32[:, :], float32[:, :, :]])
def mult_kernel(a, b, c):
    Ni = c.shape[0]
    Nj = c.shape[1]
    Nk = c.shape[2]

    startX, startY, startZ = cuda.grid(3)
    gridX = cuda.gridDim.x * cuda.blockDim.x
    gridY = cuda.gridDim.y * cuda.blockDim.y
    gridZ = cuda.gridDim.z * cuda.blockDim.z

    for i in range(startX, Ni, gridX):
        for j in range(startY, Nj, gridY):
            c[i, j] = 0
            for k in range(startZ, Nk, gridZ):
                c[i, j] = c[i, j] + mult_gpu(a[i, k], b[j, k])


def main():
    A = np.ones((20, 50000), dtype=np.float32)
    B = np.ones((3072, 50000), dtype=np.float32)
    C = np.ones((20, 3072, 50000), dtype=np.float32)
    (Ni, Nj, Nk) = C.shape

    my_gpu = cuda.get_current_device()
    thread_ct = 8
    block_ct_x = int(math.ceil(float(Ni) / thread_ct))
    block_ct_y = int(math.ceil(float(Nj) / thread_ct))
    block_ct_z = int(math.ceil(float(Nk) / thread_ct))

    blockdim = thread_ct, thread_ct, thread_ct
    griddim = block_ct_x, block_ct_y, block_ct_z
    print("Threads per block:", blockdim)
    print("Blocks per grid:", griddim)

    start = timer()
    Cg = cuda.to_device(C)
    mult_kernel[griddim, blockdim](A, B, Cg)
    Cg.to_host()
    dt = timer() - start
    print("Computation done in %f s" % (dt))

    print('C[:3,1,1] = ', C[:3, 1, 1])
    print('C[-3:,1,1] = ', C[-3:, 1, 1])


if __name__ == '__main__':
    main()
