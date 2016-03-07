import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
from scipy import scipy.stats.poisson

x= np.arange(0,10,0.1)
pois = poisson.pdf(x,5)

fig = plt.figure()
ax1 = fig.add_subplot(111) 
ax1.plot(x,pois)
plt.show()
plt.savefig('test.pdf')
