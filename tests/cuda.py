import numpy as np
from pylab import imshow, show
from timeit import default_timer as timer
from numba import autojit, prange, jit, cuda
import numba

@cuda.autojit()
def mandel(x, y, max_iters):
    """
      Given the real and imaginary parts of a complex number,
      determine if it is a candidate for membership in the Mandelbrot
      set given a fixed number of iterations.
    """
    c = complex(x, y)
    z = 0.0j
    for i in prange(max_iters):
        z = z * z + c
        if (z.real * z.real + z.imag * z.imag) >= 4:
            return i

    return max_iters

@cuda.autojit()
def create_fractal(min_x, max_x, min_y, max_y, image, iters):
    height = image.shape[0]
    width = image.shape[1]

    pixel_size_x = (max_x - min_x) / width
    pixel_size_y = (max_y - min_y) / height

    for x in prange(width):
        real = min_x + x * pixel_size_x
        for y in prange(height):
            imag = min_y + y * pixel_size_y
            color = mandel(real, imag, iters)
            image[y, x] = color


image = np.zeros((1024, 1536), dtype=np.uint8)
start = timer()
create_fractal(-2.0, 1.0, -1.0, 1.0, image, 20)
dt = timer() - start

print("Mandelbrot created in " + str(dt) + " s")
imshow(image)
show()
