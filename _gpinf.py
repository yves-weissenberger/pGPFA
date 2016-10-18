import scipy.optimize as op
import numpy as np
from _util import make_Cbig, make_Kbig, make_xbar, make_ybar, makeCd_from_vec,make_vec_Cd
from _lapinf import lap_post_unNorm, lap_post_grad, lap_post_hess
from _paraminf import Cd_obsCost, Cd_obsCost_grad

def precompute_gp(params,lapInfres):

    n_timePoints = params['latent_traj'][0].shape[1]
    nDims = params['latent_traj'][0].shape[0]
    T = len(params['t']) 
    t1 = np.tile(params['t'],[T,1])

    tdif = t1 - t1.T
    difSq = tdif**2

    precomp = []

    for dim in range(nDims):
        tempSum = 0
        for trl_idx in range(params['nTrials']):
            tempSum += (params['post_cov_GP'][trl_idx][dim] +
                        np.outer(params['latent_traj'][trl_idx][dim],
                                params['latent_traj'][trl_idx][dim])
                        )

        precomp.append( {'T':n_timePoints,
                         'Tdif': tdif,
                         'difSq': difSq,
                         'numTrials': params['nTrials'],
                         'PautoSum': tempSum
                        }
                      )
    return precomp




def GP_timescale_Cost(tav,precomp):
    tav = np.exp(tav) + 1e-3
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

