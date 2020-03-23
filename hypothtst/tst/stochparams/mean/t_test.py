import numpy as np
from scipy.stats import t, ttest_ind, norm


def t_test(mu1,mu2,std1,std2,n1,n2):
    std_dev = (std1**2/(n1-1)+std2**2/(n2-1))**0.5
    test_stat = (mu2-mu1)/std_dev
    df = n1+n2-2
    if test_stat>0:
        p_val = 2*t.sf(test_stat,df)
    else:
        p_val = 2*t.cdf(test_stat,df)
    return p_val


def tst():
    a1 = norm.rvs(10,2,size=100)
    a2 = norm.rvs(11,2.4,size=100)
    mu1, mu2 = np.mean(a1), np.mean(a2)
    std1, std2 = np.std(a1), np.std(a2)
    n1,n2 = len(a1), len(a2)
    p_val1 = ttest_ind(a1,a2)[1]
    p_val2 = t_test(mu1,mu2,std1,std2,n1,n2)
    return p_val1 == p_val2


