import numpy as np
import time

from numba import vectorize, cuda


@vectorize(["float32(float32, float32)"], target='cuda')
def VectorAdd(a, b):
    return a + b


def main():
    N = 32000000

    A = np.ones(N, dtype=np.float32)
    B = np.ones(N, dtype=np.float32)

    start = time.time()
    C = VectorAdd(A, B)
    vector_add_time = time.time() - start

    print("C[:5] = " + str(C[:5]))
    print("C[-5:] = " + str(C[-5:]))

    print("VectorAdd took for %f seconds" % vector_add_time)


if __name__ == '__main__':
    main()


    # https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1604&target_type=deblocal
    # https://devblogs.nvidia.com/parallelforall/even-easier-introduction-cuda/
    # https://developer.nvidia.com/cuda-toolkit
    # https://developer.nvidia.com/how-to-cuda-python
    # https://docs.anaconda.com/anaconda/install/linux
    # https://www.anaconda.com/download/#linux