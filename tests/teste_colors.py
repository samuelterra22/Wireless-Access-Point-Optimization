
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import *
from matplotlib.mlab import bivariate_normal

cor = colors.LogNorm(vmin=0.0, vmax=100.0)


from pylab import *

cmap = cm.get_cmap('plasma', 20)    # PiYG

print(matplotlib.colors.rgb2hex(cmap(22)))

for i in range(cmap.N):
    rgb = cmap(i)[:3] # will return rgba, we take only first 3 so we get rgb
    print(matplotlib.colors.rgb2hex(rgb))