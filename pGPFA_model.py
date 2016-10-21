
import numpy as _np

class pGPFA(object):
        

    def __init__(self,data,t,nDims,initParams=False,epsNoise=1e-3,**kwargs):



        kwargs.setdefault('cross_validate',True) 
        kwargs.setdefault('type','tvt')
        kwargs.setdefault('CV_fractions',[.7,.2,.1])
        kwargs.setdefault('seed',None)
        self.cross_validation = kwargs
        self.CV_inf = {}
        self.data = data
        self.train_data=data
        self.was_fit = False
        nTrials = len(self.train_data)
        self.n_neurons = self.train_data[0].shape[0]
        self.n_timePoints = self.train_data[0].shape[1]
        self.nDims = nDims
        self.t = t
        self.nTrials = nTrials
        if not initParams:
            self._init_fit_params()
        if self.cross_validation['cross_validate']:
            self._divide_data(seed=kwargs['seed'])
        self.params['epsNoise'] = epsNoise
        self.params['t'] = t
        self.params['logL_store'] = []
        self.params['nTrials'] = len(self.train_data)
        print 'initialised! :)'

    def _divide_data(self,seed=None):

        if self.cross_validation['type']=='k-fold':
            nTrials_train = _np.round(self.cross_validation['CV_fractions'][0]*self.nTrials)
            nTrials_CV = _np.round(self.cross_validation['CV_fractions'][1]*self.nTrials)
        elif self.cross_validation['type']=='tvt':
            nTrials_train =  int(_np.round(self.cross_validation['CV_fractions'][0]*self.nTrials))
            nTrials_validation = int(_np.round(self.cross_validation['CV_fractions'][1]*self.nTrials))
            nTrials_test = int(_np.round(self.cross_validation['CV_fractions'][2]*self.nTrials))
            trlIdxs = range(self.nTrials)
            if seed:
                _np.random.seed(0)
            _np.random.shuffle(trlIdxs)
            train_idxs = trlIdxs[:nTrials_train]
            validation_idxs = trlIdxs[nTrials_train:nTrials_train+nTrials_validation]
            test_idxs = trlIdxs[nTrials_train+nTrials_validation:]
            self.train_data = [self.data[i] for i in train_idxs]
            self.validation_data = [self.data[i] for i in validation_idxs]
            self.test_data = [self.data[i] for i in test_idxs]
            self.cross_validation['train_idxs'] = train_idxs
            self.cross_validation['validation_idxs'] = validation_idxs
            self.cross_validation['test_idxs'] = test_idxs
            
            self.dsets = {'train': self.train_data,
                          'validation': self.validation_data,
                          'test': self.test_data
                          }

    def _get_trl_by_idx(abs_trlIdx):
        print 'soon'     
    
    
    def get_abs_idx(self,dset,idx):
        return self.cross_validation[dset+'_idxs'][idx]

    def _init_fit_params(self):
        self.params = {'latent_traj': [_np.zeros([self.nDims,self.n_timePoints]) for i in range(self.nTrials)],
                       'C': _np.random.normal(size=[self.n_neurons,self.nDims]),
                       'd': _np.random.randn(self.n_neurons),
                       'l': [-1]*self.nDims
                       }

    def check_nDims(self,minD=1,maxD=18):
        print "WARNING THIS IS GONNA TAKE A LOOOOOONG TIME"
        ans = raw_input( "sure you're ready?:  (y/n)   ")

        if ans=='y':
            print 'here we go!'
        else:
            print 'aborted'

        

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
        
        self.was_fit = True

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
            self.params['logL_store'].append(Cdinf['logL'])

            if iterN>10:
                if (self.params['logL_store'][-1]>self.params['logL_store'][-2]):
                    print 'warning logL started increasing'
                    #break

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

    def test_cross_val(self,trl_type='validation'):
        import EM
        import copy as cp
        CV_params = cp.deepcopy(self.params)
        CV_params['nTrials'] = len(self.dsets[trl_type])
        CV_params['latent_trajctory'] =[_np.zeros([self.nDims,self.n_timePoints]) for i in range(len(CV_params['latent_traj']))]
        lapinfres = EM.E_step(self.dsets[trl_type],CV_params)
        self.CV_inf[trl_type] = lapinfres
        self.CV_inf[trl_type]['latent_traj'] = lapinfres['post_mean']




    def leave_n_out(self,n,trl_idx,dset='validation'):
        nIdxs = range(self.n_neurons)
        lo_idxs = _np.random.choice(nIdxs,replace=False,size=N)
        C_lo = np.delete(self.params['C'],lo_idxs,axis=0)
        d_lo = np.delete(self.params['d'],lo_idxs)
        
        y_lo = np.delete(self.dsets[dset][trl_idx])





###############################################################################
###################### Plotting Functions #####################################
###############################################################################
    def plot_trial_rates(self,trl_idx,dset='train',ground_truth=0):
        
        import seaborn
        import matplotlib.pyplot as plt
        if type(dset)==str:
            pass
        
        if dset =='train':
            x = self.params['latent_traj']
        elif dset=='validation':
            x = self.CV_inf[dset]['latent_traj']
            

        
        plt.figure(figsize=(22,18))
        clrs = seaborn.color_palette('RdBu',n_colors=6)
        inf_rates = _np.exp(self.params['C'].dot(x[trl_idx]) + self.params['d'][:,None])
        nRows = nCols = _np.ceil(_np.sqrt(self.n_neurons))
        for idx,neuron_rate in enumerate(inf_rates):

            plt.subplot(nRows,nCols,idx+1)
            obsRate = self.dsets[dset][trl_idx][idx]
            l1, = plt.plot(self.t,obsRate,color=clrs[-2])
            l2, = plt.plot(self.t,neuron_rate,color=clrs[0],linewidth=3)
            
            if ground_truth:
               gt_idx = self.get_abs_idx(idx=trl_idx,dset=dset)
               l3, =  plt.plot(self.t,ground_truth[gt_idx][idx],linestyle='--',color='k',linewidth=2)

            for hl in _np.linspace(0,1000,num=201):
                if (hl<=15 or (5+_np.max(obsRate))>hl):
                    plt.plot(self.t,[hl]*self.n_timePoints,color=[.3]*3,alpha=.2)
                    if hl>0:
                        plt.text(_np.max(self.t)-2,hl,str(hl)+'Hz',
                                color=[.3]*3,fontsize=14,alpha=.8)

            plt.xticks([],[])
            plt.yticks([],[])
            
            if idx==0:
                plt.plot([_np.min(self.t),_np.min(self.t)+2],[-2,-2],color='k',linewidth=2)
                plt.text(_np.min(self.t),-5,'2s',fontsize=18)
            
            plt.xlim([_np.min(self.t),_np.max(self.t)])
        plt.tight_layout(pad=0,w_pad=.5,h_pad=1)
        
        if ground_truth:
            plt.figlegend((l1,l2,l3),('Inferred Rate','Observed Rate','True Rate'),
            loc = 'lower center',ncol=3,frameon=True,framealpha=.8,labelspacing=36)
        else:
            plt.figlegend((l1,l2),('Inferred Rate','Observed Rate'),
                    loc = 'lower center',ncol=3,frameon=True,framealpha=.8,labelspacing=64)




    def plot_covariances(self):
        print 1



    def _sample_based_cov_mtx(self,dset):
        if dset=='validation':
            x = np.hstack(gpfa.CV_inf['validation']['latent_traj'])
        elif dset=='train':
            x = np.hstack(self.params['latent_traj'])

        infRates = np.exp(gpfa.params['C'].dot(x)+gpfa.params['d'][:,None])

        maxiter=10000

        for it in range(max_iter):
            sampInfRates = np.random.poisson(infRates)
            if it==0:
                corrMtx_fit = np.corrcoef(sampInfRates)

               




