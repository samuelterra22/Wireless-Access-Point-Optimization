import numpy as np
import pylab as pl
from math import log10
from scipy.stats import lognorm
import matplotlib.pyplot as plt

coleta = [-34,-43,-35,-48,-27,-30,-39,-40,-45,-44,-58,-53,-56,-56,-49,-49,-53,-62,-54,-66,-56,-60,-66,-56,-51,-65,-54,-59,-60,-64,-53,-69,-54,-71,-71,-58,-61,-61,-56,-56,-71]

x = np.linspace(1, 10, 200)

sigma=np.std(coleta)
mu=np.mean(coleta)
dist = lognorm(sigma, loc=mu)

pl.plot(dist.pdf(x))
#pl.plot(x)

pl.show()