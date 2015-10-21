#! /usr/bin/env python
#
# author: Timothy C Arlen
#         tca3@psu.edu
#
# date:   2015-June-04
#


import numpy as np
from scipy.optimize import curve_fit
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


# Fit a gaussian to the function:
def gauss_fn(x,*p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))

def get_bin_centers(edges):
    return (np.array(edges[:-1]) + np.array(edges[1:]))/2

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("mu1",type=float, help="Calculated mean of the first gaussian")
parser.add_argument("sigma1",type=float,
                    help="Calculated std dev of the first gaussian")
parser.add_argument("mu2",type=float, help="Calculated mean of the second gaussian")
parser.add_argument("sigma2",type=float,
                    help="Calculated std dev of the second gaussian")
parser.add_argument("trials", type=int,
                    help="Number of samples or trials in the analysis.")
parser.add_argument("--n_repeat",type=int,default=30000,
                    help='''Number of times to repeat the estimation at the number of''')
args = parser.parse_args()

true_sigma = np.fabs(args.mu2 - args.mu1)/args.sigma2
print("True significance for these parameters: ",true_sigma)

sigma_error = np.zeros(args.n_repeat)
for trial in xrange(1,args.n_repeat+1):
    dist1 = np.random.normal(args.mu1, args.sigma1, args.trials)
    dist2 = np.random.normal(args.mu2, args.sigma2, args.trials)

    calc_sigma = np.fabs(dist2.mean() - dist1.mean())/dist2.std()

    sigma_error[trial-1] = (true_sigma - calc_sigma)


print "\n\n>>>>> Results <<<<<"
print "For %d trials, for two gaussians with parameters: "%args.trials
print "  (mu1, sigma1): = ("+str(args.mu1)+", "+str(args.sigma1)+")"
print "  (mu2, sigma2): = ("+str(args.mu2)+", "+str(args.sigma2)+")"
print "The error on the significance is: ",sigma_error.std()
print "The fractional error on the significance is: ",(sigma_error.std()/true_sigma)
