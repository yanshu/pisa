import numpy as np
import sys
from sklearn import svm
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn import linear_model
import matplotlib.pyplot as plt
import h5py
from sklearn.ensemble import GradientBoostingRegressor


def plot_resolution(x, x_t, **kwargs):
    bins = np.linspace(0,200,100)
    #bins = np.linspace(-2,4,100)
    print '%.4f +/- %.4f'%(np.median((x-x_t)/x_t),np.std((x-x_t)/x_t))
    n, bins, patches = plt.hist((x), linewidth=0,
            alpha=0.5, log=False, bins=bins, **kwargs)   


fnames = ['/Users/peller/PSU/data/DC12_nutau.hd5','/Users/peller/PSU/data/DC12_numu.hd5','/Users/peller/PSU/data/DC12_nue.hd5']
files = []
for fname in fnames:
    files.append(h5py.File(fname,'r'))

idx = 100000

rng = np.random.RandomState(1)
DT = DecisionTreeRegressor(max_depth=4)
#BDT = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4)
SVR = svm.SVR(kernel='rbf', C=1., gamma=0.05)
GBR = GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, max_depth=4, random_state=0, loss='lad')
lin = linear_model.LinearRegression()
lasso = linear_model.Lasso(alpha = 0.1)

varnames = {}
varnames['reco_e'] = ['IC86_Dunkman_L6_MultiNest8D_Neutrino','energy']
varnames['reco_cz'] = ['IC86_Dunkman_L6_MultiNest8D_Neutrino','zenith']
varnames['reco_a'] = ['IC86_Dunkman_L6_MultiNest8D_Neutrino','azimuth']
varnames['reco_x'] = ['IC86_Dunkman_L6_MultiNest8D_Neutrino','x']
varnames['reco_y'] = ['IC86_Dunkman_L6_MultiNest8D_Neutrino','y']
varnames['reco_z'] = ['IC86_Dunkman_L6_MultiNest8D_Neutrino','z']
varnames['delta_LLH'] = ['IC86_Dunkman_L6','delta_LLH']
varnames['phi_prime'] = ['IC86_Dunkman_L6','phi_prime']
varnames['r_prime'] = ['IC86_Dunkman_L6','r_prime']
varnames['rho_prime'] = ['IC86_Dunkman_L6','rho_prime']
varnames['theta_prime'] = ['IC86_Dunkman_L6','theta_prime']
varnames['x_prime'] = ['IC86_Dunkman_L6','x_prime']
varnames['y_prime'] = ['IC86_Dunkman_L6','y_prime']
varnames['track_e'] = ['IC86_Dunkman_L6_MultiNest8D_Track','energy']
varnames['cscd_e'] = ['IC86_Dunkman_L6_MultiNest8D_Cascade','energy']

# construct X (input vars)
X = []
for var, dir in varnames.items():
    l = []
    for file in files:
        l.append(np.array(file[dir[0]][dir[1]][:idx]))
    X.append(np.ravel(np.array(l)))
 
X = np.array(X)
     
# construct y (target)
true_e = []
for file in files:
    true_e.append(np.array(file['trueNeutrino']['energy'][:idx]))

true_e = np.ravel(np.array(true_e))

reco_e = []
for file in files:
    reco_e.append(np.array(file['IC86_Dunkman_L6_MultiNest8D_Neutrino']['energy'][:idx]))

reco_e = np.ravel(np.array(reco_e))

#true_e = np.array(file['trueNeutrino']['energy'])
#reco_e = np.array(file['IC86_Dunkman_L6_MultiNest8D_Neutrino']['energy'])
#reco_cz = np.array(file['IC86_Dunkman_L6_MultiNest8D_Neutrino']['zenith'])
#reco_a = np.array(file['IC86_Dunkman_L6_MultiNest8D_Neutrino']['azimuth'])
#reco_x = np.array(file['IC86_Dunkman_L6_MultiNest8D_Neutrino']['x'])
#reco_y = np.array(file['IC86_Dunkman_L6_MultiNest8D_Neutrino']['y'])
#reco_z = np.array(file['IC86_Dunkman_L6_MultiNest8D_Neutrino']['z'])
#delta_LLH = np.array(file['IC86_Dunkman_L6']['delta_LLH'])
#phi_prime = np.array(file['IC86_Dunkman_L6']['phi_prime'])
#r_prime = np.array(file['IC86_Dunkman_L6']['r_prime'])
#rho_prime = np.array(file['IC86_Dunkman_L6']['rho_prime'])
#theta_prime = np.array(file['IC86_Dunkman_L6']['theta_prime'])
#x_prime = np.array(file['IC86_Dunkman_L6']['x_prime'])
#y_prime = np.array(file['IC86_Dunkman_L6']['y_prime'])
#z_prime = np.array(file['IC86_Dunkman_L6']['z_prime'])
#santa_direct_charge = np.array(file['IC86_Dunkman_L6']['santa_direct_charge'])
#track_e = np.array(file['IC86_Dunkman_L6_MultiNest8D_Track']['energy'])
#cscd_e = np.array(file['IC86_Dunkman_L6_MultiNest8D_Cascade']['energy'])
#X = np.array([reco_e, cscd_e, track_e, reco_a, reco_cz, delta_LLH])
#X = np.array([reco_e, cscd_e, track_e, reco_a, reco_cz, reco_x, reco_y, reco_z,
#        delta_LLH])
#y = true_e
out = GBR.fit(X.T[1:idx:2], true_e[1:idx:2])
new_e = out.predict(X.T[0:idx:2])
plot_resolution(reco_e[0:idx:2],true_e[0:idx:2], facecolor='blue')
plot_resolution(new_e,true_e[0:idx:2], facecolor='green')
print out.feature_importances_
plt.show()
