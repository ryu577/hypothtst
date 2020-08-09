import numpy as np
from scipy.stats import binom_test


def critical_interval(ts1,w,delt):
    t_end_prev=0
    critical=0
    for t1 in ts1:
        t_start=min(t1,t1+w)
        t_start=max(t_start,0)
        t_end=max(t1,t1+w)
        t_end=min(t_end,delt)
        t_start=max(t_start,t_end_prev)
        critical+=(t_end-t_start)
        t_end_prev=t_end
    return critical


def critical_events(ts1,ts2,w):
    j=0; critical=0
    for t in ts2:
        while j<len(ts1) and t>ts1[j]:
            j+=1
        if w>0 and j>0:
            critical+=ts1[j-1]+w>t
        elif j<len(ts1) and w<0:
            critical+=ts1[j]+w<t
    return critical


def correlation_score(ts1,ts2,w,delt,verbose=False):
    interv = critical_interval(ts1,w,delt)
    evnts = critical_events(ts1,ts2,w)
    if verbose:
        print(str(evnts)+","+str(len(ts2))+","+str(interv/delt))
    return binom_test(evnts,len(ts2),interv/delt,alternative='greater')


########################################
## Functional tests

def tst_critical_interv():
    ##### For critical interval.
    res = critical_interval([1,2,3],1,4)
    print(res==3)
    res = critical_interval([1,3],-5,4)
    print(res==3)
    res = critical_interval([1,3],-5,4)
    print(res==3)
    res = critical_interval([1,3],1,4)

def tst_critical_evnts():
    ##### For critical events.
    res = critical_events([1,2,3],[.5,1.5,2.5],.5)
    print(res==0)
    res = critical_events([1,2,3],[.5,1.5,2.5],1)
    print(res==2)
    res = critical_events([1,2,3],[.5,1.5,2.5],-1)
    print(res==3)

