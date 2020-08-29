import numpy as np
import matplotlib.pyplot as plt
from stochproc.point_processes.renewal.lomax import lomax_renewal_stats, p_vals_lomax_renewal
from algorith.arrays.pt_process.overlap import critical_events
from algorith.arrays.pt_process.window import critical_interval
from hypothtst.alpha_beta_sim import rejectn_rate
from hypothtst.viz_utils.plots_on_pvals import plot_power_3alt, plot_4_pvals


def get_lomax_renewal_split_p_vals():
    p_vals_nul = p_vals_lomax_renewal(theta=3,k=3,null=True,window=.1)
    #(theta:1,corln:0.807);(2,0.6);(3,0.44);(4,0.37);(5,0.28) 
    p_vals_alt1 = p_vals_lomax_renewal(theta=1,k=1/1.5+1,window=.1)
    p_vals_alt2 = p_vals_lomax_renewal(theta=2,k=2/1.5+1,window=.1)
    p_vals_alt3 = p_vals_lomax_renewal(theta=3,k=3,window=.1)
    plot_4_pvals(p_vals_nul, p_vals_alt1,p_vals_alt2,p_vals_alt3)
    plot_power_3alt(p_vals_nul, p_vals_alt1,p_vals_alt2,p_vals_alt3)


def which_lomax_renewal_split_better():
    theta=7
    ## Smaller rains upon larger.
    p_vals_nul1 = p_vals_lomax_renewal(theta=theta,k=theta/1.5+1,null=True,window=10,split_p=0.3,n_sim=10000)
    p_vals_alt1 = p_vals_lomax_renewal(theta=theta,k=theta/1.5+1,null=False,window=10,split_p=0.3,n_sim=10000)
    ## Larger rains upon smaller.
    p_vals_nul2 = p_vals_lomax_renewal(theta=theta,k=theta/1.5+1,null=True,window=-10,split_p=0.7,n_sim=10000)
    p_vals_alt2 = p_vals_lomax_renewal(theta=theta,k=theta/1.5+1,null=False,window=-10,split_p=0.7,n_sim=10000)
    plt.hist(p_vals_nul1,alpha=0.5,label="null1", histtype='step',\
                    fill=False,stacked=True,color='green')
    plt.hist(p_vals_nul2,alpha=0.5,label="null2", histtype='step',\
                    fill=False,stacked=True,color='yellow')
    plt.hist(p_vals_alt1,alpha=0.5,label="alt1", histtype='step',\
                    fill=False,stacked=True,color='red')
    plt.hist(p_vals_alt2,alpha=0.5,label="alt2", histtype='step',\
                    fill=False,stacked=True,color='orange')
    plt.legend()
    plt.show()
    alpha1 = rejectn_rate(p_vals_nul1)
    power1 = rejectn_rate(p_vals_alt1)
    alpha2 = rejectn_rate(p_vals_nul2)
    power2 = rejectn_rate(p_vals_alt2)
    plt.plot(alpha1,power1,label="smaller_on_larger")
    plt.plot(alpha2,power2,label="larger_on_smaller")
    plt.xlabel("False positive rate")
    plt.ylabel("Power (true negative rate)")
    plt.legend()
    plt.show()

