import numpy as np
from _util import makeCd_from_vec
def Cd_obsCost(vecCd,ys,x,postCov,params):
    """ Calculate the log-likelihood of parameters given observations current estimate 
        of the latent states and the obervations
        
        "Estimating state and parameters in state space models of spike trains"
        Macke, Buesing and Sahani (2015) in Advanced State Space Methods for Neural
        and Clinical Data p149 """
    
    
    LL = 0
    n_timePoints = x[0].shape[1]
    nDims = x[0].shape[0]
    n_neurons = ys[0].shape[0]

    C,d = makeCd_from_vec(vecCd,nDims,n_neurons)
    C = C
    for trl_idx in range(params['nTrials']):
        y = ys[trl_idx]
        for i in range(n_timePoints):
            t1 = C.dot(x[trl_idx][:,i]) + d
            t2 = .5*np.diag(C.dot(postCov[trl_idx][i]).dot(C.T))
            LL += y[:,i].dot(t1) - np.dot(np.ones(n_neurons),np.exp(t1+ t2))
    return np.divide(-LL,params['nTrials'])
def Cd_obsCostFast(vecCd,ys,x,postCov,params):
    """ Entirely stolen from Hooram Nam's 2015 
        implementation 

        https://github.com/mackelab/poisson-gpfa/blob/master/funs/learning.py
    """
    n_timePoints = x[0].shape[1]
    nDims = x[0].shape[0]
    n_neurons = ys[0].shape[0]
    
    C,d = makeCd_from_vec(vecCd,nDims,n_neurons)
    CC = np.zeros([n_neurons,nDims**2])


    for dim in range(n_neurons):
        CC[dim,:] = np.reshape(np.outer(C[dim,:],C[dim,:]),nDims**2)

    f = 0
    for trl_idx in range(params['nTrials']):
        y = ys[trl_idx]
        vsm = np.reshape(params['post_cov_Cd'][trl_idx],[n_timePoints,nDims**2])
        hh = np.dot(C,params['latent_traj'][trl_idx]) + d[:,None]
        rho = np.dot(CC,vsm.T)
        yhat = np.exp(hh+rho/2)
        f += f + np.sum(np.sum(y*hh - yhat))

    return np.divide(-f,params['nTrials'])



def Cd_obsCost_grad(vecCd,ys,x,postCov,params):
    
    LL = 0
    n_timePoints = x[0].shape[1]
    nDims = x[0].shape[0]
    n_neurons = ys[0].shape[0]


    C,d = makeCd_from_vec(vecCd,nDims,n_neurons)
    C_grad = np.zeros(C.shape)
    d_grad = np.zeros(d.shape)
    
    for trl_idx in range(params['nTrials']):
        y = ys[trl_idx]
        for i in range(n_timePoints):

            yhat = np.exp(C.dot(x[trl_idx][:,i]) + d + .5*np.diag(C.dot(postCov[trl_idx][i]).dot(C.T)))
            term2 = np.dot(np.diag(yhat),C.dot(postCov[trl_idx][i]))
            C_grad += np.dot(np.array([y[:,i]-yhat]).T,np.array([x[trl_idx][:,i]])) - term2
            d_grad += y[:,i] - yhat
    return np.divide( -np.concatenate([C_grad.T.flatten(),d_grad]),params['nTrials'])
