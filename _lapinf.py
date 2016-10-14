
import numpy as np
import scipy as sp
def lap_post_unNorm(xbar, ybar, C_big, d,  K_bigInv,t,n_neurons):


    """ calculate p(y|x;0)""" 
    nT = len(t)
    xbar = np.array([np.squeeze(xbar)]).T
    #print xbar.shape
    A = np.squeeze(C_big.dot(xbar)) + np.tile(d,nT)
    
    L1 = np.dot(np.ones(n_neurons*nT),np.exp(A))
    L2 =  ybar.dot(A)
    L3 =  .5*xbar.T.dot(K_bigInv.dot(xbar))
    p = L1 - L2 + L3 
    return p

def lap_post_grad(xbar, ybar, C_big, d,  K_bigInv,t,n_neurons):
    """ calculate dp(y|x;0)/dx"""
    nT = len(t)
    xbar = np.array([np.squeeze(xbar)]).T
    A = np.squeeze(C_big.dot(xbar)) + np.tile(d,nT)

    dL1 = np.dot(np.exp(A),C_big)
    dL2 = np.dot(ybar, C_big)
    dL3 = np.dot(xbar.T,  K_bigInv)

    dL = dL1 - dL2 + dL3

    return np.squeeze(dL)


def lap_post_hess(xbar, ybar, C_big, d, K_bigInv,t,n_neurons):

    

    """ calculate dp(y|x;0)2/dx2"""
    nT = len(t)
    xbar = np.array([np.squeeze(xbar)]).T
    A = np.squeeze(C_big.dot(xbar)) + np.tile(d,nT)

    Aexpdiagonal = sp.sparse.spdiags(np.exp(A),0,n_neurons*nT,n_neurons*nT)
    temp = Aexpdiagonal.dot(C_big)

    ddL = np.dot(C_big.T, temp) +  K_bigInv
    
    return ddL
