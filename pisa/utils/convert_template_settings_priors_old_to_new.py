#!/usr/bin/env python 

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '--infile',
    type=str,
    required=True
)
parser.add_argument(
    '--outfile',
    type=str,
    required=True
)
args = parser.parse_args()

import sys, os, re, traceback, time, warnings, itertools
import copy
from pisa.utils import utils as putils
from pisa.utils import params as ppars

ts0 = putils.from_file(args.infile)
ts1 = copy.deepcopy(ts0)
for paramname, param in sorted(ts0['params'].iteritems()):
    new_prior = ppars.Prior.from_old_style_param_dict(param_dict=param)
    if new_prior is None:
        continue
    new_param = copy.deepcopy(param)
    new_param.update(new_prior.build_dict())
    ts1['params'][paramname] = new_param

putils.to_file(ts1, args.outfile)
