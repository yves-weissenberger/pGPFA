
import numpy as _np

class pGPFA(object):
        

    def __init__(self,data,t,nDims,initParams=False,epsNoise=1e-3):
        self.data = data

        nTrials = len(self.data)
        self.n_neurons = self.data[0].shape[0]
        self.n_timePoints = self.data[0].shape[1]
        self.nDims = nDims
        self.t = t
        self.nTrials = nTrials
        if not initParams:
            self._init_params()
            
        self.params['epsNoise'] = epsNoise
        self.params['t'] = t
        self.params['nTrials'] = nTrials
        print 'initialised! :)'

    def _init_params(self):
        self.params = {'latent_traj': [_np.zeros([self.nDims,self.n_timePoints]) for i in range(self.nTrials)],
                       'C': _np.random.normal(size=[self.n_neurons,self.nDims]),
                       'd': _np.random.randn(self.n_neurons),
                       'l': [-1]*self.nDims
                       }


    def fit(self,nIter=20):
        import time
        import EM
        from _gpinf import precompute_gp, GP_timescale_Cost
        import scipy.optimize as op

        st = time.time()
        self.hists = {'lapinf':[],
                      'Cdinf':[],
                      'tavinf':[]
                     }
        
        
        for iterN in range(nIter):
            print "Running EM iteration %s" %iterN,
            #######   E-step   ###########
            lapinfres = EM.E_step(self.data,self.params)
            self._update_params_E(lapinfres)

            self.hists['lapinf'].append(lapinfres)
            #######   M-step   ###########
            Cdinf, tavInf = EM.M_step(self.data,self.params)

            self.hists['Cdinf'].append(Cdinf); self.hists['tavinf'].append(tavInf)

            self._update_params_M(Cdinf,tavInf)
            
            print "|| log(L) after M step is: %s ||  total time elapsed: %ss" %(_np.round(Cdinf['logL'],decimals=2),_np.round(time.time()-st,decimals=1))

    def _update_params_E(self,lapinfres):
        self.params['post_cov_Cd'] = lapinfres['post_cov_Cd']
        self.params['post_cov_GP'] = lapinfres['post_cov_GP']
        self.params['latent_traj'] = lapinfres['post_mean']

    def _update_params_M(self,Cdinf,tavInf):
        self.params['C'] = Cdinf['Cinf']
        self.params['d'] = Cdinf['dinf']

        for dim in range(self.nDims):
           self.params['l'][dim] = (1/_np.exp(tavInf[dim].x))**(0.5)
           #print self.params['l'][dim]

            
