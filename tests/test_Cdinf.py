import sys
import os
import numpy as np
from numpy.linalg import norm
import scipy.optimize as op
import matplotlib.pyplot as plt
import seaborn
from statsmodels.tools.numdiff import approx_hess
seaborn.set(font_scale=2)
seaborn.set_style('whitegrid')

sys.path.append('/home/yves/Documents')
import pGPFA
#from pGPFA import genSim_data_static
from pGPFA._paraminf import Cd_obsCost, Cd_obsCost_grad
from pGPFA._util import make_vec_Cd, makeCd_from_vec, make_xbar

def test_Cd_grads():
    
    """ Function to test the accuracy of the analytical gradient of
        the cost function of observations with respect to C, the 
        loading matrix, and d, the baseline firing rates 
    """

    y,params,t = pGPFA.genSim_data_static(n_neurons=40,
                                    nDims=3,
                                    pretty=False
                                    )

    lapInfres = pGPFA.E_step(y,params)
    #Cd_obsCost(vecCd,n_neurons,nDims,nT,y,x,postCov) 
    C = params['C']; d = params['d']
    Cd_opt = make_vec_Cd(C,d) 
    Cd_rand = np.random.normal(size=Cd_opt.shape)
    #xbar = make_xbar(lapInfres['post_mean'] 
    n_timePoints = params['latent_traj'].shape[1]
    nDims = params['latent_traj'].shape[0]

    cov_store = np.zeros([n_timePoints,nDims,nDims])
    for tk in range(n_timePoints):
        cov_store[tk][:,:] = lapInfres['post_cov'][tk*nDims:(1+tk)*nDims,tk*nDims:(1+tk)*nDims]
    res_approxCd = op.approx_fprime(Cd_opt,
                                    Cd_obsCost,
                                    .5e-5,
                                    y,
                                    lapInfres['post_mean'],
                                    cov_store
                                    )

    trueCd = Cd_obsCost_grad(Cd_opt,y,lapInfres['post_mean'],cov_store)
    diff = norm(res_approxCd -trueCd)/norm(res_approxCd+trueCd)
    #print("Normed difference between approximate and analytical gradients is
    #%s" %diff )
    print diff
    plt.plot(res_approxCd)
    plt.plot(trueCd)
    plt.show()
    return None

if __name__=="__main__":
    print('testing gradients for C and d...')
    test_Cd_grads()


