import numpy as np
from scipy.stats import binom_test
import matplotlib.pyplot as plt
from hypothtst.alpha_beta_sim import AlphaBetaSim
import hypothtst.tst.stochparams.mean.t_test as ttst
from scipy.stats import ttest_ind
from scipy.stats import poisson, norm, t
import matplotlib.pyplot as plt
from hypothtst.sim_utils.mean.normal import NormDist

def tst_simultn():
    #po0=PoissonDist(5)
    #po1=PoissonDist(10)
    po0 = poisson(5); po1 = poisson(10)
    t0=5; t1=10
    ## A hypothesis test that takes the two samples as arguments.
    tst = lambda n0,n1:binom_test(n1,n0+n1,t1/(t0+t1),\
                    alternative='greater')
    ab = AlphaBetaSim()
    alphas,betas=ab.alpha_beta_tracer(po0,po1,tst,tst,n_sim=10000)
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

def beta_plots_poisson_on_poisson():
    t0=10; t1=3
    po0=poisson(2*t0); #po0.random_state = np.random.RandomState(seed=20)
    po1=poisson(3*t1); #po1.random_state = np.random.RandomState(seed=50)
    ## A hypothesis test that takes the two samples as arguments.
    tst_1 = lambda n0,n1:binom_test(n1,n0+n1,t1/(t0+t1),\
                    alternative='greater')
    tst_1_null = lambda n0,n1:binom_test(n1,n0+n1,0.5,\
                    alternative='greater')
    tst_2 = lambda n0,n1:binom_test(n1/3,n0/3+n1/3,t1/(t0+t1),\
                    alternative='greater')
    tst_2_null = lambda n0,n1:binom_test(n1/3,n0/3+n1/3,0.5,\
                    alternative='greater')
    tst_3 = lambda n0,n1:binom_test(n1/.3,n0/.3+n1/.3,t1/(t0+t1),\
                    alternative='greater')
    tst_3_null = lambda n0,n1:binom_test(n1/.3,n0/.3+n1/.3,0.5,\
                    alternative='greater')
    tst_4 = lambda n0,n1:binom_test(n1,n0+n1,t1/(t0+t1),\
                    alternative='two-sided')
    tst_4_null = lambda n0,n1:binom_test(n1,n0+n1,0.5,\
                    alternative='two-sided')
    tst_5 = lambda n0,n1:binom_test(n1/3,n0/3+n1/3,t1/(t0+t1),\
                    alternative='two-sided')
    tst_5_null = lambda n0,n1:binom_test(n1/3,n0/3+n1/3,0.5,\
                    alternative='two-sided')
    tst_6 = lambda n0,n1:binom_test(n1/2,n0/2+n1/2,t1/(t0+t1),\
                    alternative='two-sided')
    tst_6_null = lambda n0,n1:binom_test(n1/2,n0/2+n1/2,0.5,\
                    alternative='two-sided')
    tst_7 = lambda n0,n1:binom_test(n1/.1,n0/.1+n1/.1,t1/(t0+t1),\
                    alternative='two-sided')
    tst_7_null = lambda n0,n1:binom_test(n1/.1,n0/.1+n1/.1,0.5,\
                    alternative='two-sided')
    ab1 = AlphaBetaSim()
    alphas1,betas1=ab1.alpha_beta_tracer(po0,po1,tst_1_null,tst_1,n_sim=10000)
    print("first sim")
    alphas2,betas2=ab1.alpha_beta_tracer(po0,po1,tst_2_null,tst_2,n_sim=10000)
    print("second sim")
    alphas3,betas3=ab1.alpha_beta_tracer(po0,po1,tst_3_null,tst_3,n_sim=10000)
    print("third sim")
    alphas4,betas4=ab1.alpha_beta_tracer(po0,po1,tst_4_null,tst_4,n_sim=10000)
    print("fourth sim")
    alphas5,betas5=ab1.alpha_beta_tracer(po0,po1,tst_5_null,tst_5,n_sim=10000)
    print("fifth sim")
    alphas6,betas6=ab1.alpha_beta_tracer(po0,po1,tst_6_null,tst_6,n_sim=10000)
    print("sixth sim")
    alphas7,betas7=ab1.alpha_beta_tracer(po0,po1,tst_7_null,tst_7,n_sim=10000)
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


def beta_plots_t_on_gaussian(n0=10, n1=12):
    g0 = NormDist(10, 3, n0)
    g1 = NormDist(11, 2.5, n1)
    t_tst_obj0 = ttst.TTest(alternative='two-sided')
    tst_0 = t_tst_obj0.tst
    t_tst_obj1 = ttst.TTest(alternative='greater')
    tst_1 = t_tst_obj1.tst
    #
    gaus = ttst.Norm_null()
    t_tst_obj3 = ttst.TTest(alternative='two-sided', dist=gaus)
    tst_2 = t_tst_obj3.tst
    t_tst_obj4 = ttst.TTest(alternative='greater', dist=gaus)
    tst_3 = t_tst_obj4.tst
    ##
    gaus1 = ttst.Norm_null(1,2)
    t_tst_obj4 = ttst.TTest(alternative='two-sided',dist=gaus1)
    tst_4 = t_tst_obj4.tst
    t_tst_obj5 = ttst.TTest(alternative='greater',dist=gaus1)
    tst_5 = t_tst_obj5.tst
    ##
    cauch1 = ttst.Cauchy_null()
    t_tst_obj6 = ttst.TTest(alternative='two-sided',dist=cauch1)
    tst_6 = t_tst_obj6.tst
    t_tst_obj7 = ttst.TTest(alternative='greater',dist=cauch1)
    tst_7 = t_tst_obj7.tst
    ab = AlphaBetaSim()
    #alphas1,betas1=ab.alpha_beta_tracer(g0,g1,tst_0,tst_0,n_sim=10000)
    #print("sim1 done")
    alphas2,betas2=ab.alpha_beta_tracer(g0,g1,tst_1,tst_1,n_sim=10000)
    print("sim2 done")
    #alphas3,betas3=ab.alpha_beta_tracer(g0,g1,tst_2,tst_2,n_sim=10000)
    #print("sim3 done")
    alphas4,betas4=ab.alpha_beta_tracer(g0,g1,tst_3,tst_3,n_sim=10000)
    print("sim4 done")
    #alphas5,betas5=ab.alpha_beta_tracer(g0,g1,tst_4,tst_4,n_sim=10000)
    #print("sim5 done")
    alphas6,betas6=ab.alpha_beta_tracer(g0,g1,tst_5,tst_5,n_sim=10000)
    print("sim6 done")
    #alphas7,betas7=ab.alpha_beta_tracer(g0,g1,tst_6,tst_6,n_sim=10000)
    #print("sim7 done")
    alphas8,betas8=ab.alpha_beta_tracer(g0,g1,tst_7,tst_7,n_sim=10000)
    print("sim8 done")
    #plt.plot(alphas1,betas1,label='t_alternate_2sided')
    plt.plot(alphas2, betas2, label='t_null_distribution')
    #plt.plot(alphas3,betas3,label='normal_alternate_2sided')
    plt.plot(alphas4, betas4, label='standard_normal_null_distribution')
    #plt.plot(alphas5,betas5,label='norm_non_std_alternate_greater')
    plt.plot(alphas6,betas6,label='non_standard_normal_null_distribution')
    #plt.plot(alphas7,betas7,label='norm_non_std_alternate_greater')
    plt.plot(alphas8,betas8,label='cauchy_null_dstribution')
    plt.legend(prop={'size': 20})
    plt.xlabel("False positive rate")
    plt.ylabel("False negative rate")
    plt.show()


def same_var_diff_var_t_test(ax, n_prms0=(10, 15, 26),
                             n_prms00=(10, 5, 6),
                             n_prms1=(13, 5, 6)):
    g0 = NormDist(*n_prms0)
    g00 = NormDist(*n_prms00)
    g1 = NormDist(*n_prms1)
    t_tst_obj0 = ttst.TTest_diffvar(alternative='two-sided')
    tst_0 = t_tst_obj0.tst
    t_tst_obj1 = ttst.TTest_equalvar(alternative='two-sided')
    tst_1 = t_tst_obj1.tst
    ab = AlphaBetaSim()
    #alphas1, betas1 = ab.alpha_beta_tracer(g0, g1, tst_0, n_sim=10000)
    #alphas2, betas2 = ab.alpha_beta_tracer(g0, g1, tst_1, n_sim=10000)
    #ax.plot(alphas1, betas1, label="Different variance test")
    #ax.plot(alphas2, betas2, label="Equal variance test")
    alphas1, betas1 = ab.alpha_beta_tracer2(g0,g00,g1,tst_0)
    alphas2, betas2 = ab.alpha_beta_tracer2(g0,g00,g1,tst_1)
    #ax.plot(ab.alpha_hats, alphas1, label="Different variance test")
    #ax.plot(ab.alpha_hats, alphas2, label="Equal variance test")
    ax.plot(alphas1, betas1, label="Different variance test")
    ax.plot(alphas2, betas2, label="Equal variance test")
    #ax.xlabel("Simulated false positive rate")
    #ax.ylabel("Simulated false negative rate")
    ax.legend()


def plot_grid():
    ns = np.array([0.3, 1.0, 5.0])
    sigs = np.array([0.3, 1.0, 3.0])
    fig, axs = plt.subplots(len(ns),len(sigs))
    i=-1
    for n_ratio in ns:
        i+=1
        n1 = int(30)
        n2 = int(n1*n_ratio)
        j=-1
        for sig_ratio in sigs:
            j+=1
            sig1 = 10
            sig2 = sig1*sig_ratio
            prms0 = (10, sig1, n1)
            prms00 = (10, sig2, n2)
            prms1 = (13, sig2, n2)
            ax = axs[i, j]
            ax.set_title('n1='+str(n1)+' n2=' + str(n2)+' sig1='+str(sig1)+' sig2='+str(sig2))
            same_var_diff_var_t_test(ax, prms0, prms00, prms1)
            print("Plotted: " + str((i,j)))
    plt.show()


def demo_welch_worse():
    a1 = norm.rvs(10, 14, size=6)
    a2 = norm.rvs(13, 3, 100)
    p_val1 = ttest_ind(a1, a2)[1]
    p_val2 = ttest_ind(a1, a2, equal_var=False)[1]
    print("Same var p-val:" + str(p_val1))
    print("Different var p-val:" + str(p_val2))

    cnt1=0; cnt2=0
    for _ in range(10000):
        a1 = norm.rvs(10, 14, size=6)
        a2 = norm.rvs(10, 3, 100)
        p_val1 = ttest_ind(a1, a2)[1]
        p_val2 = ttest_ind(a1, a2, equal_var=False)[1]
        cnt1+=(p_val1<0.05)
        cnt2+=(p_val2<0.05)


def fnr_vs_sample_size(sample_size=(10, 10), n_sim=10000, sig_tau=0.05):
    cnt = 0
    for _ in range(n_sim):
        a1 = norm.rvs(10, 3, size=sample_size[0])
        a2 = norm.rvs(13, 3, size=sample_size[1])
        p_val1 = ttest_ind(a1, a2)[1]
        cnt += (p_val1 > sig_tau)
    fnr = cnt/n_sim
    return fnr


def get_curve():
    fnrs = []
    for size in np.arange(3, 100):
        fnr = fnr_vs_sample_size((size, size))
        fnrs.append(fnr)
    plt.plot(np.arange(3, 100), fnrs)
    plt.show()
    return fnrs
