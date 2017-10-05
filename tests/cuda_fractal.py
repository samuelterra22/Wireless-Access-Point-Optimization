from timeit import default_timer as timer
from numba import cuda
from numba import *
from timeit import default_timer as timer

from numba import *
from numba import cuda


# @guvectorize(["void(float32, float32, uint8, uint8[:])"], "(),(),()->()", target='cuda')
# def mandel(x, y, max_iters, out):
# @guvectorize(['uint8(float32, float32, uint8, uint8[:])'], '(),(),()->()', target='cuda')
# def mandel(x, y, max_iters, out):
@autojit
def mandel(x, y, max_iters):
    """
      Given the real and imaginary parts of a complex number,
      determine if it is a candidate for membership in the Mandelbrot
      set given a fixed number of iterations.
    """
    # out[0] = 0

    c = complex(x, y)
    z = 0.0j
    # i = np.dtype(0, np.uint8)
    for i in range(max_iters):
        z = z * z + c
        if (z.real * z.real + z.imag * z.imag) >= 4:
            # out[0] = i
            return i

    # if out > 0:
    #    out[0] = max_iters
    return max_iters


@cuda.jit
def mandel_kernel(min_x, max_x, min_y, max_y, image, iters):
    height = image.shape[0]
    width = image.shape[1]

    pixel_size_x = (max_x - min_x) / width
    pixel_size_y = (max_y - min_y) / height

    ## CUDA FIX
    startX, startY = cuda.grid(2)
    gridX = cuda.gridDim.x * cuda.blockDim.x
    gridY = cuda.gridDim.y * cuda.blockDim.y

    ## CUDA FIX
    for x in range(startX, width, gridX):
        # for x in range(width):
        real = min_x + x * pixel_size_x
        ## CUDA FIX
        for y in range(startY, height, gridY):
            # for y in range(height):
            imag = min_y + y * pixel_size_y
            image[y, x] = mandel_gpu(real, imag, iters)


# @guvectorize(["void(float32, float32, float32, float32, int32, uint8[:,:])"], "(),(),(),(),()->(,)", target='cuda')
# def create_fractal(min_x, max_x, min_y, max_y, iters, image):
# @guvectorize(["void(uint8[:,:],uint8[:])"], "(m,n)", target='cuda')
# def create_fractal(image,out):

mandel_gpu = cuda.jit(device=True)(mandel)

gimage = np.zeros((1024 * 10, 1536 * 10), dtype=np.uint8)

## CUDA WTF
## A GPU GeForce GT 740M tem 384 núcleos
## blocos de 48x8 = 384 threads
blockDim = (48, 8)
## cada thread trabalha num grid de 32x16
gridDim = (32, 16)

start = timer()

## copia a matrix g_image da RAM para a GPU (d_image)
d_image = cuda.to_device(gimage)

## calcula a matrix d_image na GPU, passando para cada thread os blocos
mandel_kernel[gridDim, blockDim](-2.0, 1.0, -1.0, 1.0, d_image, 20)

## copia o resultado da d_image GPU para g_image na RAM
d_image.to_host()

dt = timer() - start

print("Mandelbrot created in " + str(dt) + " s")
# imshow(gimage)
# show()
