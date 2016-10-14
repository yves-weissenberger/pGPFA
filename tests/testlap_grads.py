import scipy.optimize as op
from statsmodels.tools.numdiff import approx_hess1
import numpy as np
from numpy.linalg import norm
import sys
sys.path.append('/home/yves/Documents')
from pGPFA._lapinf import lap_post_unNorm, lap_post_grad, lap_post_hess
from pGPFA._simdata import genSim_data_static
from pGPFA._util import *
import matplotlib.pyplot as plt
import seaborn
seaborn.set(font_scale=2)
seaborn.set_style('whitegrid')

def test_grad():
    
    y,params,t = genSim_data_static()
    ybar = make_ybar(y)
    n_timePoints = len(t)
    x_opt = params['latent_traj']
    n_neurons = y.shape[0]
    x_init = np.random.random(size=x_opt.shape)
    C_big = make_Cbig(params['C'],n_timePoints)
    d = params['d']
    nDims = x_opt.shape[0]
    K_big,_ = make_Kbig(params,t,nDims)
    xbar = make_xbar(x_opt)
    K_big += np.eye(K_big.shape[0])*1e-3
    K_bigInv = np.linalg.inv(K_big)
    res = lap_post_grad(np.squeeze(xbar),ybar,C_big,d,K_bigInv,t,n_neurons)
    res2 = op.approx_fprime(np.squeeze(xbar),
                     lap_post_unNorm,
                     1e-6,
                     ybar,C_big,d,K_bigInv,t,n_neurons
                     )
   
    print ("\n norm of error in gradient calculdation is %s"
            %np.divide(norm(res-res2),norm(res+res2))
          )
    plt.figure()
    plt.plot(res2,label='numerical approximation')
    plt.plot(res,label='analytical gradient')
    plt.legend()
    plt.ylabel('gradient')
    plt.xlabel('xidx')
    plt.show()
    return None


if __name__=="__main__":
    import numpy as np
    test_grad()

