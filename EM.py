import numpy as np
import scipy.optimize as op
from _util import make_Cbig, make_Kbig, make_xbar, make_ybar, makeCd_from_vec,make_vec_Cd, covByDim
from _lapinf import lap_post_unNorm, lap_post_grad, lap_post_hess
from _paraminf import Cd_obsCost, Cd_obsCost_grad, Cd_obsCostFast,Cd_obsCost_gradFast
from _gpinf import precompute_gp, GP_timescale_Cost





def E_step(ys,params,alpha=0):

    n_timePoints = ys[0].shape[1] 
    n_neurons = ys[0].shape[0]
    C = params['C']; d = params['d']
    C_big = make_Cbig(C,n_timePoints)
    
    nDims = C.shape[1]

    K_big,_ = make_Kbig(params,params['t'],nDims,epsNoise=1e-3)
    K_bigInv = np.linalg.inv(K_big+ np.eye(K_big.shape[0])*1e-3)
    lapRes = []
    for trl_idx in range(params['nTrials']):
        
        x = params['latent_traj'][trl_idx]
        xbar = make_xbar(x)
        ybar = make_ybar(ys[trl_idx])

        resLap = op.minimize(
            fun = lap_post_unNorm,
            x0 = x,
            method='Newton-CG',
            args = (ybar, C_big, d, K_bigInv,params['t'],n_neurons,alpha),
            jac = lap_post_grad,
            hess = lap_post_hess,
            options = {'disp': False,'maxiter': 500,'xtol':1e-16
            })
        x_post_mean = resLap.x.reshape(nDims,n_timePoints,order='F')
        postCov = np.linalg.inv(lap_post_hess(resLap.x,ybar, C_big, d,
                                K_bigInv,params['t'],n_neurons,alpha)
                               )
        postL = resLap.fun
        #for inference of C and d
        post_cov_by_timepoint  = np.zeros([n_timePoints,nDims,nDims])
        #for inference of C and d
        for i in range(n_timePoints):
                post_cov_by_timepoint[i] = postCov[i*nDims:(i+1)*nDims,i*nDims:(i+1)*nDims]

        #for inference of tav of the GP
        postCov_GP, post_cov_by_latent = covByDim(postCov,nDims,n_timePoints)
        if trl_idx==0: 
            lapInfRes = {'post_mean': [x_post_mean],
                     'post_cov': [postCov],
                     'post_cov_Cd':[post_cov_by_timepoint],
                     'post_cov_GP':[post_cov_by_latent],
                     'post_cov_alt':[postCov_GP],
                     'logL': [postL] 
                    }
        else:
            lapInfRes['post_mean'].append(x_post_mean)
            lapInfRes['post_cov'].append(postCov)
            lapInfRes['post_cov_Cd'].append(post_cov_by_timepoint)
            lapInfRes['post_cov_GP'].append(post_cov_by_latent)
            lapInfRes['post_cov_alt'].append(postCov_GP)
            lapInfRes['logL'].append(postL)
        #lapRes.append(lapInfRes)


    return lapInfRes



def E_step(ys,params,alpha=0):

    n_timePoints = ys[0].shape[1] 
    n_neurons = ys[0].shape[0]
    C = params['C']; d = params['d']
    C_big = make_Cbig(C,n_timePoints)
    
    nDims = C.shape[1]

    K_big,_ = make_Kbig(params,params['t'],nDims,epsNoise=1e-3)
    K_bigInv = np.linalg.inv(K_big+ np.eye(K_big.shape[0])*1e-3)
    lapRes = []
    for trl_idx in range(params['nTrials']):
        
        x = params['latent_traj'][trl_idx]
        xbar = make_xbar(x)
        ybar = make_ybar(ys[trl_idx])

        resLap = op.minimize(
            fun = lap_post_unNorm,
            x0 = x,
            method='Newton-CG',
            args = (ybar, C_big, d, K_bigInv,params['t'],n_neurons,alpha),
            jac = lap_post_grad,
            hess = lap_post_hess,
            options = {'disp': False,'maxiter': 500,'xtol':1e-16
            })
        x_post_mean = resLap.x.reshape(nDims,n_timePoints,order='F')
        postCov = np.linalg.inv(lap_post_hess(resLap.x,ybar, C_big, d,
                                K_bigInv,params['t'],n_neurons,alpha)
                               )
        postL = resLap.fun
        #for inference of C and d
        post_cov_by_timepoint  = np.zeros([n_timePoints,nDims,nDims])
        #for inference of C and d
        for i in range(n_timePoints):
                post_cov_by_timepoint[i] = postCov[i*nDims:(i+1)*nDims,i*nDims:(i+1)*nDims]

        #for inference of tav of the GP
        postCov_GP, post_cov_by_latent = covByDim(postCov,nDims,n_timePoints)
        if trl_idx==0: 
            lapInfRes = {'post_mean': [x_post_mean],
                     'post_cov': [postCov],
                     'post_cov_Cd':[post_cov_by_timepoint],
                     'post_cov_GP':[post_cov_by_latent],
                     'post_cov_alt':[postCov_GP],
                     'logL': [postL] 
                    }
        else:
            lapInfRes['post_mean'].append(x_post_mean)
            lapInfRes['post_cov'].append(postCov)
            lapInfRes['post_cov_Cd'].append(post_cov_by_timepoint)
            lapInfRes['post_cov_GP'].append(post_cov_by_latent)
            lapInfRes['post_cov_alt'].append(postCov_GP)
            lapInfRes['logL'].append(postL)
        #lapRes.append(lapInfRes)


    return lapInfRes

def M_step(ys,params):
    n_timePoints = ys[0].shape[1] 
    n_neurons = ys[0].shape[0]
    C = params['C']; d = params['d']
    #C_big = make_Cbig(C,n_timePoints)
 
    x = params['latent_traj']

    nDims = C.shape[1]
    vecCd = make_vec_Cd(params['C'],params['d']) 
    ####Infer the C and d parameters
    resCd = op.minimize(
        fun = Cd_obsCostFast,
        x0 = vecCd,
        method = 'TNC',
        args = (ys,params['latent_traj'],params['post_cov_Cd'],params),
        jac = Cd_obsCost_gradFast,
        options = {'disp': False,
                  'maxiter':500,'ftol':1e-16,'gtol':1e-16,'xtol':1e-16}
        )
    C_inf,d_inf = makeCd_from_vec(resCd.x,nDims,n_neurons)
    Cdinfres = {'Cinf':C_inf,
                'dinf':d_inf,
                'logL':resCd.fun
               }
    ####Infer the GP timescale parameters
            
    precomp = precompute_gp(params)
    tavInf = []
    for dim in range(nDims):
        res = op.minimize(GP_timescale_Cost,
                          x0 = params['l'][dim],
                          args=(precomp[dim],params),
                          method='TNC',
                          options={'minfev':0,'gtol':1e-8,'eps':1e-8}
                          )

        tavInf.append(res)

    return Cdinfres, tavInf
