import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from matplotlib.ticker import LogFormatter

delta = 0.025

x = y = np.arange(0, 3.01, delta)
X, Y = np.meshgrid(x, y)
Z1 = plt.mlab.bivariate_normal(X, Y, 1.0, 1.0, 0.0, 0.0)
Z2 = plt.mlab.bivariate_normal(X, Y, 1.5, 0.5, 1, 1)
Z = 10 * (Z1 * Z2)

fig = plt.figure()
ax1 = fig.add_subplot(111)
lvls = np.logspace(-4, 0, 20)
CF = ax1.contourf(X, Y, Z,
                  norm=LogNorm(),
                  levels=lvls
                  )
CS = ax1.contour(X, Y, Z,
                 norm=LogNorm(),
                 colors='k',
                 levels=lvls
                 )

l_f = LogFormatter(10, labelOnlyBase=False)

cbar = plt.colorbar(CF, ticks=lvls, format=l_f)
plt.show()
