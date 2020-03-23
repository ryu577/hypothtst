import numpy as np
from scipy.stats import binom_test
import matplotlib.pyplot as plt
from hypothtst.alpha_beta_sim import AlphaBetaSim
from hypothtst.sim_utils.rate.poisson import PoissonDist
from scipy.stats import poisson
import matplotlib.pyplot as plt

def tst_simultn():
    po0=PoissonDist(5)
    po1=PoissonDist(10)
    t0=5; t1=10
    ## A hypothesis test that takes the two samples as arguments.
    tst = lambda n0,n1:binom_test(n1,n0+n1,t1/(t0+t1),\
                    alternative='greater')
    ab = AlphaBetaSim()
    alphas,betas=ab.alpha_beta_tracer(po0,po1,tst,n_sim=10000)
    plt.plot(alphas,betas)
    plt.show()

def p_val_uniform():
    p_vals = []
    for _ in range(1000):
        n0=poisson.rvs(100)
        n1=poisson.rvs(100)
        p_val = binom_test(n0,n1+n0,.5,alternative='greater')
        p_vals.append(p_val)
    plt.hist(p_vals)

def beta_plots():
    t0=10; t1=20
    po0=PoissonDist(1*t0)
    po1=PoissonDist(1.4*t1)
    ## A hypothesis test that takes the two samples as arguments.
    tst_1 = lambda n0,n1:binom_test(n1,n0+n1,t1/(t0+t1),\
                    alternative='greater')
    tst_2 = lambda n0,n1:binom_test(n1/3,n0/3+n1/3,t1/(t0+t1),\
                    alternative='greater')
    tst_3 = lambda n0,n1:binom_test(n1/.3,n0/.3+n1/.3,t1/(t0+t1),\
                    alternative='greater')
    tst_4 = lambda n0,n1:binom_test(n1,n0+n1,t1/(t0+t1),\
                    alternative='two-sided')
    tst_5 = lambda n0,n1:binom_test(n1/3,n0/3+n1/3,t1/(t0+t1),\
                    alternative='two-sided')
    tst_6 = lambda n0,n1:binom_test(n1/2,n0/2+n1/2,t1/(t0+t1),\
                    alternative='two-sided')
    tst_7 = lambda n0,n1:binom_test(n1/.1,n0/.1+n1/.1,t1/(t0+t1),\
                    alternative='two-sided')
    ab1 = AlphaBetaSim()
    alphas1,betas1=ab1.alpha_beta_tracer(po0,po1,tst_1,n_sim=10000)
    print("first sim")
    alphas2,betas2=ab1.alpha_beta_tracer(po0,po1,tst_2,n_sim=10000)
    print("second sim")
    alphas3,betas3=ab1.alpha_beta_tracer(po0,po1,tst_3,n_sim=10000)
    print("third sim")
    alphas4,betas4=ab1.alpha_beta_tracer(po0,po1,tst_4,n_sim=10000)
    print("fourth sim")
    alphas5,betas5=ab1.alpha_beta_tracer(po0,po1,tst_5,n_sim=10000)
    print("fifth sim")
    alphas6,betas6=ab1.alpha_beta_tracer(po0,po1,tst_6,n_sim=10000)
    print("sixth sim")
    alphas7,betas7=ab1.alpha_beta_tracer(po0,po1,tst_7,n_sim=10000)
    print("seventh sim")
    plt.plot(alphas1,betas1,label='sc:1_greater')
    plt.plot(alphas2,betas2,label='sc:3_greater')
    plt.plot(alphas3,betas3,label='sc:.3_greater')
    plt.plot(alphas4,betas4,label='sc:1_two_sided')
    plt.plot(alphas5,betas5,label='sc:3_two_sided')
    plt.plot(alphas6,betas6,label='sc:2_two_sided')
    plt.plot(alphas7,betas7,label='sc:.1_two_sided')
    plt.legend()
    plt.show()

