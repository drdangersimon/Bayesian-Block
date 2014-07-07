import numpy as np
import pylab as lab
from scipy imports stats


###priors###
def geomentric_prior(gamma, N_blocks, N):
    '''Geometric prior. Weights more for fewer number of blocks'''
    return ((1 - gamma) * gamma**N_blocks) / ((1 - gamma)**(N + 1))


####likelyhoods



####main
