## https://gist.github.com/hawkerpl/5487c62d6b14b3f8b40d

import numpy as np

def _range_when_zero(iterlist):
    for i, pixel in enumerate(iterlist):
        if pixel == 0:
            return i
    else:
        return float(len(iterlist))


def _count_step(d_x, d_y, x, y, i, sina, minim):
    if minim == 1:
        return np.floor(sina*i), i
    else:
        return i, np.floor(i/sina)

def _range_when_non_zero(d_x, d_y, x, y, tab):
    maxval = max(d_x, d_y)
    sina = d_x/float(d_y)
    minim = (d_x, d_y).index(maxval)
    for i in xrange(maxval):
        step_x, step_y = _count_step(d_x, d_y, x, y, i, sina, minim)
        value = tab[y+step_y, x+step_x, 0]
        if value == 0:
            return np.sqrt(step_x**2 + step_y**2)
    return np.sqrt(d_x**2 + d_y**2)

def trace_ray(tab, start, end):
    x0, y0 = map(int,start)
    x1, y1 = map(int,end)
    d_x = x1 - x0
    d_y = y1 - y0
    if d_x == 0:
        ym = (y0,y1)
        return _range_when_zero(iterlist=tab[min(ym):max(ym), x0, 0])
    elif d_y == 0:
        xm = (x0,x1)
        return _range_when_zero(iterlist=tab[y0, min(xm):max(xm), 0])
    else:
        return _range_when_non_zero(d_x, d_y, x0, y0, tab)