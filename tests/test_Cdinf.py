import sys
import os
import numpy as np
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
from pGPFA._util import make_vec_Cd, makeCd_from_vec

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
    res_approx = op.approx_fprime(Cd_rand,
                                  Cd_obsCost,
                                  1e-4,
                                  y,
                                  lapInfres['post_mean'],
                                  lapInfres['post_cov']
                                  )
    return None

if __name__=="__main__":
    print('testing gradients for C and d...')
    test_Cd_grads()


