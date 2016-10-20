
from __future__ import division
import matplotlib.pyplot as plt
import seaborn
seaborn.set_style('whitegrid')
seaborn.set(font_scale=2)


def plot_latent_states(self,trl_idx,show_error=True,nRows=4):
    clrs = seaborn.color_palette(n_colors=self.nDims)

    if self.nDims<=6:
        self.nRows=2
    for dim in range(self.nDims):
        plt.subplot(nRows,np.ceil(self.nDims/nRows)
        x = gpfa.params['latent_traj'][trl_idx][dim].T
        std2 = np.sqrt(np.diag(self.params['post_cov_GP'][trl_idx][i]))
        plt.plot(self.params['t'],x,color=clrs[i])

        if show_error:
            plt.fill_between(self.params['t'],x-std2,x+std2,clr=clrs[i],alpah=.4)
    return None


def plot_activity(self,trl,trl_type):

    for n in range(self.n_neurons):
        print 1
