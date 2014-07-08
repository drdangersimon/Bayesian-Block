import numpy as nu
import pylab as lab
from scipy import stats


###priors###
def geomentric_prior(gamma, N_blocks, N):
    '''Geometric prior. Weights more for fewer number of blocks'''
    return ((1 - gamma) * gamma**N_blocks) / ((1 - gamma)**(N + 1))


####likelihoods
def poisson(N,M):
    '''Function from MSc work, will look up what M is'''
    #N!*(rheman_zeta(N+1) - sum(1/(M**(N+1)))


####main
def bayesian_blocks(t,x):
    """Bayesian Blocks Implementation

    """
    # copy and sort the array
    N = t.size
    best = nu.zeros_like(t)
    last =  nu.zeros_like(t)
    #-----------------------------------------------------------------
    # Start with first data cell; add one cell at each iteration
    #-----------------------------------------------------------------
    for K in xrange(N):
        # Compute the width and count of the final bin for all possible
        # locations of the K^th changepoint
        mean, sigma = nu.mean(x[:K +1]), nu.std(x[:K +1])
        if sigma == 0:
            sigma = 1e99
        fit_vec = -((mean - x[:K + 1])/(2*sigma))**2.
        #width = block_length[:K + 1] - block_length[K + 1]
        #count_vec = nu.cumsum(nn_vec[:K + 1][::-1])[::-1]

        # evaluate fitness function for these possibilities
        #fit_vec = count_vec * (np.log(count_vec) - np.log(width))
        #fit_vec -= 4  # 4 comes from the prior on the number of changepoints
        fit_vec[1:] += best[:K]

        # find the max of the fitness: this is the K^th changepoint
        i_max = nu.argmax(fit_vec)
        last[K] = i_max
        best[K] = fit_vec[i_max]

    #-----------------------------------------------------------------
    # Recover changepoints by iteratively peeling off the last block
    #-----------------------------------------------------------------
    change_points =  nu.zeros(N, dtype=int)
    i_cp = N
    ind = N
    while True:
        i_cp -= 1
        change_points[i_cp] = ind
        if ind == 0:
            break
        ind = last[ind - 1]
    change_points = change_points[i_cp:]

    # Calculate means and times for change points
    out_mean, out_time = [] ,[]
    for i in range(1,len(change_points)):
        imin, imax = change_points[i-1], change_points[i]
        out_mean.append([x[imin:imax].mean()]*2)
        try:
            out_time.append([t[change_points[i-1]], t[change_points[i]]])
        except:
            out_time.append([t[change_points[i-1]], t[change_points[i]-1]])
    return out_time, out_mean
