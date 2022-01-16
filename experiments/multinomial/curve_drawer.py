import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import time
from hypothtst.tst.stochparams.multinomial.test_stats import get_pval_sim, get_p_val_chisq
from hypothtst.alpha_beta_sim import AlphaBeta_1Sampl


class Multinom():
    def __init__(self, k, n, ps=None):
        self.k = k
        self.n = n
        if ps is None:
            self.ps = np.ones(k)/k
        else:
            self.ps = ps

    def rvs(self):
        ys = np.random.choice(self.k, p=self.ps, size=self.n)
        ns = Counter(ys).values()
        ns = list(ns)
        return ns


def main():
    dist1 = Multinom(3, 10)
    dist2 = Multinom(3, 10, ps=[.6,.2,.2])
    ab = AlphaBeta_1Sampl()
    ab.alpha_beta_tracer(dist1, dist2, get_p_val_chisq)
    ab2 = AlphaBeta_1Sampl()
    start_time = time.time()
    ab2.alpha_beta_tracer(dist1, dist2, get_pval_sim)
    print("--- %s seconds ---" % (time.time() - start_time))
    plt.plot(ab.alphas, ab.betas)
    plt.plot(ab2.alphas, ab2.betas)
    plt.show()
