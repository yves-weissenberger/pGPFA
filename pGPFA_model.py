
import numpy as _np

class pGPFA(object):
        

    def __init__(self,data,t,nDims,initParams=False,epsNoise=1e-3,**kwargs):



        kwargs.setdefault('cross_validate',True)
        kwargs.setdefault('type','tvt')
        kwargs.setdefault('CV_fractions',[.7,.2,.1])
        self.cross_validation = kwargs

        self.data = data
        self.train_data=data

        nTrials = len(self.train_data)
        self.n_neurons = self.train_data[0].shape[0]
        self.n_timePoints = self.train_data[0].shape[1]
        self.nDims = nDims
        self.t = t
        self.nTrials = nTrials
        if not initParams:
            self._init_fit_params()
        if self.cross_validation['cross_validate']:
            self._divide_data()
        self.params['epsNoise'] = epsNoise
        self.params['t'] = t
        self.params['nTrials'] = len(self.train_data)
        print 'initialised! :)'

    def _divide_data(self):

        if self.cross_validation['type']=='k-fold':
            nTrials_train = _np.round(self.cross_validation['CV_fractions'][0]*self.nTrials)
            nTrials_CV = _np.round(self.cross_validation['CV_fractions'][1]*self.nTrials)
        elif self.cross_validation['type']=='tvt':
            nTrials_train =  int(_np.round(self.cross_validation['CV_fractions'][0]*self.nTrials))
            nTrials_validate = int(_np.round(self.cross_validation['CV_fractions'][1]*self.nTrials))
            nTrials_test = int(_np.round(self.cross_validation['CV_fractions'][2]*self.nTrials))
            trlIdxs = range(self.nTrials)
            _np.random.shuffle(trlIdxs)
            train_idxs = trlIdxs[:nTrials_train]
            validate_idxs = trlIdxs[nTrials_train:nTrials_train+nTrials_validate]
            test_idxs = trlIdxs[nTrials_train+nTrials_validate:]
            self.train_data = [self.data[i] for i in train_idxs]
            self.validate_data = [self.data[i] for i in validate_idxs]
            self.test_data = [self.data[i] for i in test_idxs]
            self.cross_validation['train_idxs'] = train_idxs
            self.cross_validation['validate_idxs'] = validate_idxs
            self.cross_validation['test_idxs'] = test_idxs
            
            self.dsets = {'train': self.train_data,
                          'validate': self.validate_data,
                          'test': self.test_data
                          }
    def _get_trl_by_idx(trlIdx):
        print 1
    def _init_fit_params(self):
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
            lapinfres = EM.E_step(self.train_data,self.params)
            self._update_params_E(lapinfres)

            self.hists['lapinf'].append(lapinfres)
            #######   M-step   ###########
            Cdinf, tavInf = EM.M_step(self.train_data,self.params)

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



    def leave_n_out(self,n,trl_idx,dset='validate'):
        nIdxs = range(self.n_neurons)
        lo_idxs = _np.random.choice(nIdxs,replace=False,size=N)
        C_lo = np.delete(self.params['C'],lo_idxs,axis=0)
        d_lo = np.delete(self.params['d'],lo_idxs)
        
        y_lo = np.delete(self.dsets[dset][trl_idx])


