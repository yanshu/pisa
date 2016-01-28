#! /usr/bin/env python
#
# BuildFisherMatrix.py
#
# Tools for building the Fisher Matrix from a set of derivatives and a fiducial
# map.
#
# author: Lukas Schulte - schulte@physik.uni-bonn.de
#         Sebastian Boeser - sboeser@uni-mainz.de
#

import numpy as np

from pisa.analysis.fisher.Fisher import FisherMatrix
from pisa.utils.params import Prior
from pisa.utils.log import logging

def build_fisher_matrix(gradient_maps, fiducial_map, template_settings):

  # fix the ordering of parameters
  params = gradient_maps.keys()
  fisher = {}
  for chan in gradient_maps[params[0]]:

    # Find non-empty bins in flattened map
    nonempty = np.nonzero(fiducial_map[chan]['map'].flatten())
    logging.info("Using %u non-empty bins of %u" %
                 (len(nonempty[0]), len(fiducial_map[chan]['map'].flatten())))

    # get gradients as calculated above for non-zero bins
    gradients = np.array(
        [ gradient_maps[par][chan]['map'].flatten()[nonempty] for par in params ]
    )
    # get error estimate from best-fit bin count for non-zero bins
    sigmas = np.sqrt(fiducial_map[chan]['map'].flatten()[nonempty])

    # Loop over all parameter per bin (simple transpose) and calculate Fisher
    # matrix by getting the outer product of all gradients in a bin. Result is
    # sum of matrix for all bins
    fmatrix = np.zeros((len(params), len(params)))
    for bin_gradients, bin_sigma in zip(gradients.T,sigmas.flatten()):
      fmatrix += np.outer(bin_gradients, bin_gradients)/bin_sigma**2
    
    # Construct the fisher matrix object
    fisher[chan] = FisherMatrix(
        matrix=fmatrix,
        parameters=params,  #order is important here!
        best_fits=[template_settings[par]['value'] for par in params],
        priors=[Prior.from_param(template_settings[par]) for par in params],
    )

  # Return all fisher matrices
  return fisher
