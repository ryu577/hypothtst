import matplotlib.pyplot as plt
from hypothtst.tst.greater.rate.ump import UMPPoisson
from scipy.special import comb
import numpy as np

def experiments():
    ##t1 and t2 are in 100-VM-days
    ### lmb_base: 1 failure per 100-VM-days.
    ## 10 nodes per hw and 10 VMs per node. So, 100 VMs per day.

    UMPPoisson.beta_on_poisson_closed_form(t1=1.0,t2=1.0,\
                            lmb_base=20,\
                            alpha=0.1,effect=20)

    ## We need 20 events per 100-VM-days.

    n=660
    UMPPoisson.beta_on_poisson_closed_form(t1=n/10,t2=n/10,\
                            lmb_base=20,\
                            alpha=0.1,effect=20*.1)


    UMPPoisson.beta_on_poisson_closed_form2(t1=1.0,t2=1.0,\
                            lmb_base=20,\
                            alpha=0.1,effect=20)

    res=UMPPoisson.beta_on_negbinom_closed_form2(t1=200,t2=200,cut_dat=1000)
    plt.plot(res[2],res[1])
    plt.axvline(res[3])
    plt.show()


def binom_partial_sum(n,p=.5):
    b_sum=0
    for j in range(int(n/1.5)+1):
        b_sum+=comb(n,j)*(1+p)**j
    return b_sum/(2+p)**n

if __name__ == '__main__':
    sums = np.array([binom_partial_sum(i,p=0.4) for i in range(11,501,2)])
    plt.plot(np.arange(11,501,2),sums)

