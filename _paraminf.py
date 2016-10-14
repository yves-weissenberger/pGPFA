import numpy as np
from _util import makeCd_from_vec
def Cd_obsCost(vecCd,y,x,postCov):
    """ Calculate the log-likelihood of parameters given observations current estimate 
        of the latent states and the obervations
        
        "Estimating state and parameters in state space models of spike trains"
        Macke, Buesing and Sahani (2015) in Advanced State Space Methods for Neural
        and Clinical Data p149 """
    
    
    LL = 0
    n_timePoints = x.shape[1]
    nDims = x.shape[0]
    n_neurons = y.shape[0]

    C,d = makeCd_from_vec(vecCd,nDims,n_neurons)
    C = C
 
    for i in range(n_timePoints):
        t1 = C.dot(x[:,i]) + d
        t2 = .5*np.diag(C.dot(postCov[i]).dot(C.T))
        LL += y[:,i].dot(t1) - np.dot(np.ones(n_neurons),np.exp(t1+ t2))
    return -LL

def Cd_obsCost_grad(vecCd,y,x,postCov):
    
    n_timePoints = x.shape[1]
    nDims = x.shape[0]
    n_neurons = y.shape[0]
    
    C,d = makeCd_from_vec(vecCd,nDims,n_neurons)
    C_grad = np.zeros(C.shape)
    d_grad = np.zeros(d.shape)
 
    for i in range(n_timePoints):

        yhat = np.exp(C.dot(x[:,i]) + d + .5*np.diag(C.dot(postCov[i]).dot(C.T)))
        term2 = np.dot(np.diag(yhat),C.dot(postCov[i]))
        C_grad += np.dot(np.array([y[:,i]-yhat]).T,np.array([x[:,i]])) - term2
        d_grad += y[:,i] - yhat
    return -np.concatenate([C_grad.T.flatten(),d_grad])
