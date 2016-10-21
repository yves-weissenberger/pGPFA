from scipy.stats import multivariate_normal
import numpy as np

from _util import calc_K



def _gen_rand_Cd(n_neurons,nDims):

	""" 
		Generate random matrices of baseline firing rates, d, and
		loadings from latent states, C
	"""

	C = np.random.normal(loc=0,scale=.5,size=(n_neurons,nDims))/2
	d = np.random.randint(0,4,size=(n_neurons))/2
	return C,d


def _gen_sample_traj(t,l,pretty,n_timePoints):

	""" 
		Generate sample trajectory from GP
		
		Arguments:
		_______________________________________

		t:         array 
				   array of size (n_timePoints,) with linear spacing

		l:		   float
				   length scale of the GP

		pretty:    bool
				   if true, trajectories have on and off ramps, looks kinda more pretty
	"""
	Ki = calc_K(x=t,y=t,l=l)
	Ki /= np.max(Ki)
	mvn1 = multivariate_normal(mean=[np.random.randint(1,5)-3]*n_timePoints,cov=Ki)
	if pretty:
		traj = mvn1.rvs()*np.concatenate([np.zeros(5),
										  np.cos(np.linspace(-np.pi/2,0,num=15)),
										  np.ones(n_timePoints-35),
										  np.cos(np.linspace(0,.5*np.pi,num=10)),
										  np.zeros(5)])
	else:
		traj = mvn1.rvs()

	return traj

def genSim_data_static(n_neurons=80,nDims=3,n_timePoints=67,pretty=False,nTrials=5):
	
    """ 
		Generate simulated data for testing the algorithm
		
		Arguments:
		_______________________________________

		n_neurons:    int
		              number of neurons to simulate

		nDims: 	      int
				      number of latent dimensions
		

	    n_timePoints: int
					  number of timepoints comrpising a trial

    """
    t = np.linspace(-33,33,num=n_timePoints)/10
    C,d =_gen_rand_Cd(n_neurons,nDims)

    length_scales_GP = [10**(1 if i== 0 else -i*.2) for i in range(nDims)]
    latent_trajs = [];
    CIFs = []
    ys = []
    for trl_idx in range(nTrials):
        x = np.zeros([nDims,n_timePoints])  #the latent trajectories
        for i in range(nDims):
		    x[i] = _gen_sample_traj(t,l=length_scales_GP[i],pretty=pretty,n_timePoints=n_timePoints) #latent states
        
        
        cifs = np.exp(C.dot(x) + d[:,None])
        y = np.random.poisson(cifs +
                np.abs(np.random.normal(loc=0,scale=1,size=cifs.shape))) #spike trains
        CIFs.append(cifs)
        ys.append(y)
        latent_trajs.append(x)
    ground_truth_dict = {'C':C,
					    'd': d,
					    'l':length_scales_GP,
					    'latent_traj': latent_trajs,
					    'CIFs': CIFs,
                        't': t,
                        'nTrials':nTrials
                        }

    return ys, ground_truth_dict




