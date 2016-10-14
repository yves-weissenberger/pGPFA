import numpy as np
import scipy.spatial.distance as dst


def make_Kbig(params,t,nDims,epsNoise=1e-3):

    epsSignal = (1-epsNoise)
    nT = len(t)  #number of time points
    #nDims = params['C'].shape[1] #dimensionality of the latent space
    #nT is
    K_big = np.zeros([nT*nDims,nT*nDims])  
    R_big = np.zeros([nT*nDims,nT*nDims]) 

    for dim in range(nDims):

        for t1_idx,t1 in enumerate(t):

            for t2_idx,t2 in enumerate(t):

                K_big[dim+t1_idx*nDims,dim+t2_idx*nDims] = epsSignal*np.exp((-0.5)*((t1-t2)**2/(params['l'][dim])**2))
                R_big[dim+t1_idx*nDims,dim+t2_idx*nDims] = 1
    return K_big, R_big




def make_Cbig(C,n_timePoints):
    """ Since C is assumed constant across all trials,
        vectorise many computations using kronecker 
        product
    """
    return np.kron(np.eye(n_timePoints),C)

def make_vec_Cd(C,d):
    """ convert between vector and matrix forms of C and d"""
    vecCd = np.vstack([C.T,d]).flatten()
    return vecCd


def makeCd_from_vec(vecCd,nDims,n_neurons):
    """ convert between vector and matrix forms of C and d"""
    C = vecCd[:nDims*n_neurons].reshape(nDims,n_neurons).T
    d = vecCd[nDims*n_neurons:]
    return C,d

def make_ybar(y):
    
    """ 
        y.shape = (n_neurons,n_timepoints).
        ybar is ordered as [neuron1_time1,neuron2_time1, ...]
    """
    ybar =  np.array([y.T.flatten()])
    return ybar

def make_xbar(x):
    xbar = x.T.flatten()
    return xbar


def get_sqdists(x,y=None):
    
    if type(y)!=np.ndarray:
        if x.ndim==1:
            dists = dst.pdist(np.vstack([x,np.zeros(x.shape)]).T,metric='sqeuclidean')
        else:
            dists = dst.pdist(x)
            
        dists = dst.squareform(dists)
        
    else:
        if x.ndim==1:
            dists = dst.cdist(np.vstack([x,np.zeros(x.shape)]).T,np.vstack([y,np.zeros(y.shape)]).T,metric='sqeuclidean')
        else:
            dists = dst.cdist(x,y)
    return dists


def calc_K(x,y=None,l=.5,add_offset=1e-3,reshape_params=None):    
    """ Calculate the covariance matrix between x and y,
        for use in a GP 

        Arguments:
        _______________________________________

        x:             array

        y:             array

        l:             float

        add_offset:    float

    """
    distsSq = get_sqdists(x,y)
    
    cov = (1-add_offset)*np.exp(-.5*distsSq/(l**2)) 
    
    cov += np.eye(len(x))*add_offset
    return cov
