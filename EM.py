import numpy as np
import scipy.optimize as op
from _util import make_Cbig, make_Kbig, make_xbar, make_ybar
from _lapinf import lap_post_unNorm, lap_post_grad, lap_post_hess

def E_step(y,params):
    x = params['latent_traj']
    nDims = x.shape[0]
    n_timePoints = x.shape[1]
    n_neurons = y.shape[0]
    K_big,_ = make_Kbig(params,params['t'],nDims,epsNoise=1e-3)
    K_bigInv = np.linalg.inv(K_big+ np.eye(K_big.shape[0])*1e-3)

    C = params['C']; d = params['d']
    C_big = make_Cbig(C,n_timePoints)

    xbar = make_xbar(x)
    ybar = make_ybar(y)

    resLap = op.minimize(
        fun = lap_post_unNorm,
        x0 = x,
        method='Newton-CG',
        args = (ybar, C_big, d, K_bigInv,params['t'],n_neurons),
        jac = lap_post_grad,
        hess = lap_post_hess,
        options = {'disp': False,'maxiter': 500,'xtol':1e-16
        })
    x_post_mean = resLap.x.reshape(nDims,n_timePoints,order='F')
    postCov = np.linalg.inv(lap_post_hess(resLap.x,ybar, C_big, d,
                            K_bigInv,params['t'],n_neurons)
                           )
    postL = resLap.fun
    lapInfRes = {'post_mean': x_post_mean,
                 'post_cov': postCov,
                 'logL': postL 
                }
    return lapInfRes


#def inference()
