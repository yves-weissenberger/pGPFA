import scipy.optimize as op
import numpy as np
from _util import make_Cbig, make_Kbig, make_xbar, make_ybar, makeCd_from_vec,make_vec_Cd
from _lapinf import lap_post_unNorm, lap_post_grad, lap_post_hess
from _paraminf import Cd_obsCost, Cd_obsCost_grad

def precompute_gp(params,lapInfres):

    n_timePoints = params['latent_traj'].shape[1]
    nDims = params['latent_traj'].shape[0]
    T = len(params['t']) 
    t1 = np.tile(params['t'],[T,1])

    tdif = t1 - t1.T
    difSq = tdif**2

    precomp = []
    for dim in range(nDims):
        tempSum = lapInfres['post_cov_GP'][dim] + np.outer(lapInfres['post_mean'][dim],lapInfres['post_mean'][dim])

        precomp.append( {'T':n_timePoints,
                         'Tdif': tdif,
                         'difSq': difSq,
                         'numTrials': 1,
                         'PautoSum': tempSum
                        }
                      )
    return precomp




def GP_timescale_Cost(tav,precomp):
    tav = np.exp(tav)
    n_timePoints = precomp['T']

    temp1 = (1-1e-3)*np.exp(-precomp['difSq']*tav*.5)
    K = temp1 + 1e-3*np.eye(n_timePoints)
    s,logdet = np.linalg.slogdet(K)
    logdet_K = logdet*s
    Kinv = np.linalg.inv(K)

    Kinv_vec = np.reshape(Kinv,n_timePoints**2)
    tempSum_vec = np.reshape(precomp['PautoSum'],n_timePoints**2)

    f = .5*logdet_K + .5*np.dot(tempSum_vec,Kinv_vec)
    return f

