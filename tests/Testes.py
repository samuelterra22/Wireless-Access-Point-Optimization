"""
https://matplotlib.org/users/pyplot_tutorial.html
http://reliawiki.org/index.php/The_Lognormal_Distribution
http://www.boost.org/doc/libs/1_43_0/libs/math/doc/sf_and_dist/html/math_toolkit/dist/dist_ref/dists/lognormal_dist.html
https://docs.scipy.org/doc/scipy/reference/tutorial/stats.html


"""

import numpy as np
import pylab as pl
from math import log10
from scipy.stats import lognorm
import matplotlib.pyplot as plt


#stddev = 0.859455801705594
#mean = 0.418749176686875
#dist = lognorm([stddev], loc=mean)



def mw_to_dbm(mW):
    """This function converts a power given in mW to a power given in dBm."""
    return 10.*log10(mW)

def dbm_to_mw(dBm):
    """This function converts a power given in dBm to a power given in mW."""
    return 10**((dBm)/10.)

coleta = [-34,-43,-35,-48,-27,-30,-39,-40,-45,-44,-58,-53,-56,-56,-49,-49,-53,-62,-54,-66,-56,-60,-66,-56,-51,-65,-54,-59,-60,-64,-53,-69,-54,-71,-71,-58,-61,-61,-56,-56,-71]
itu = [-1.3436E+02, -1.4279E+02, -1.4772E+02, -1.5122E+02, -1.5393E+02, -1.5615E+02, -1.5802E+02, -1.5965E+02, -1.6108E+02, -1.6236E+02, -1.6352E+02, -1.6458E+02, -1.6555E+02, -1.6645E+02, -1.6729E+02, -1.6808E+02, -1.6881E+02, -1.6951E+02, -1.7017E+02, -1.7079E+02, -1.7138E+02, -1.7195E+02, -1.7249E+02, -1.7301E+02, -1.7350E+02, -1.7398E+02, -1.7444E+02, -1.7488E+02, -1.7531E+02, -1.7572E+02, -1.7612E+02, -1.7651E+02, -1.7688E+02, -1.7724E+02, -1.7760E+02, -1.7794E+02, -1.7827E+02, -1.7860E+02, -1.7891E+02, -1.7922E+02, -1.7952E+02]
friis = [-3.1898E-02, -7.9744E-03, -3.5442E-03, -1.9936E-03, -1.2759E-03, -8.8605E-04, -6.5097E-04, -4.9840E-04, -3.9380E-04, -3.1898E-04, -2.6362E-04, -2.2151E-04, -1.8874E-04, -1.6274E-04, -1.4177E-04, -1.2460E-04, -1.1037E-04, -9.8450E-05, -8.8359E-05, -7.9744E-05, -7.2330E-05, -6.5904E-05, -6.0298E-05, -5.5378E-05, -5.1036E-05, -4.7186E-05, -4.3755E-05, -4.0686E-05, -3.7928E-05, -3.5442E-05, -3.3192E-05, -3.1150E-05, -2.9291E-05, -2.7593E-05, -2.6039E-05, -2.4612E-05, -2.3300E-05, -2.2090E-05, -2.0972E-05, -1.9936E-05, -1.8975E-05]
doisRaios = [37, 25, 18, 13, 9, 6, 3, 1, -1, -3, -4, -6, -7, -9, -10, -11, -12, -13, -14, -15, -16, -17, -17, -18, -19, -19, -20, -21, -21, -22, -22, -23, -24, -24, -25, -25, -26, -26, -26, -27, -27]

# teste de visualizacao da lognormal: OK
#x = np.linspace(1, 10, 200)
sigma=np.std(coleta)
mu=np.mean(coleta)
dist = lognorm(sigma, loc=mu)
#pl.plot(x, dist.pdf(x))
#pl.plot(x, dist.cdf(x))
#pl.plot(x)

pl.plot(coleta)

coleta_mW = []
for value in coleta:
    coleta_mW.append( dbm_to_mw(value) )

logNormal_mW = np.random.lognormal( np.mean(coleta_mW) , np.std(coleta_mW) , len(coleta))
#pl.plot(logNormal_mW)


logNormal_dBm = []
for value in logNormal_mW:
    logNormal_dBm.append( mw_to_dbm(value) )

#pl.plot(logNormal_dBm)
#logNormal_dBm.sort()
#pl.plot( logNormal_dBm[::-1] )

logNormal_dBm_GAMBIARRA = []
#for value in logNormal_dBm[::-1]:
for value in logNormal_dBm:

    logNormal_dBm_GAMBIARRA.append( value*10000 - 70 )

pl.plot(logNormal_dBm_GAMBIARRA)

#for i in range(1, 30):
#    logNormal.append( np.random.lognormal(mu, sigma) )
#    #logNormal.append(np.random.lognormal(-10.47, 1.6))




#pl.plot(itu)
#pl.plot(friis)
#pl.plot(doisRaios)
pl.show()

#mu, sigma = np.mean(coleta_mW), np.std(coleta_mW) # mean and standard deviation
#s = np.random.lognormal(mu, sigma, 100)
##print(s)
##pl.plot(s)
#s_dBm = []
#for value in s:
#    s_dBm.append(mw_to_dbm(value))
#print(logNormal_dBm)
#pl.plot(s_dBm)
#count, bins, ignored = plt.hist(s, 100, normed=True, align='mid')
#x = np.linspace(min(bins), max(bins), 10000)
#pdf = (np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2)) / (x * sigma * np.sqrt(2 * np.pi)))
#plt.plot(x, pdf, linewidth=2, color='r')
#plt.axis('tight')
#plt.show()