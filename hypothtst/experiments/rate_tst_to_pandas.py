"""
Python sample code for applying the 
rate test to each row
of a pandas dataframe.
"""
import numpy as np
import pandas as pd
from scipy.stats import binom_test

def apply_binom_tst(n2,t2,sc=1.0):
    t1 = 100
    n1 = 100*10
    return binom_test(n2/sc,n1/sc+n2/sc,t2/(t1+t2),alternative='greater')

def air_rank(data=None):
    if data is None:
        data = {'n':[500,6000,130,12], 't':[20, 21, 19, 18]}
        df = pd.DataFrame(data)
    df["p_val"] = df.apply(lambda x:apply_binom_tst(x[0],x[1]),axis=1)
    return df


## Mirror: https://msazure.visualstudio.com/One/_git/Compute-Insights-optimum_settings?path=%2Fcentral_lib%2Fhypothtst%2Ffor_kusto_plugin%2Fapply_binom_tst.py&version=GBmaster

