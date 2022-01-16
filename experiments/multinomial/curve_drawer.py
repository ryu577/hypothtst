import numpy as np
import matplotlib.pyplot as plt
from hypothtst.tst.stochparams.multinomial.test_stats import get_pval_sim, get_p_val_chisq


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
		return ys


class AlphaBeta_1Sampl(object):
    def __init__(self, alpha_hats=None):
        if alpha_hats is not None:
            self.alpha_hats = alpha_hats
        else:
            self.alpha_hats = np.concatenate((np.arange(\
                0.000000000001,0.0099,0.0000001),
                np.arange(0.01,1.00,0.001),
                np.arange(0.991,1.00,0.001)),axis=0)

    def alpha_beta_tracer(self,null,alter,tst_null,tst_alt=None,n_sim=10000):
        if tst_alt is None:
            tst_alt = tst_null
        self.alphas = self.rejection_rate(null,tst_null,n_sim)
        self.betas = 1-self.rejection_rate(alter,tst_alt,n_sim)

    def rejection_rate(self,dist1,tst,n_sim=10000):
        """
        At what rate is the null hypothesis that some property of two
        distributions is the same getting rejected?
        """
        rejectn_rate = np.zeros(len(self.alpha_hats))
        # First generate from null and find alpha_hat and alpha.
        for _ in range(n_sim):
            m1 = dist1.rvs()
            p_val = tst(m1)
            rejectn_rate += (p_val < self.alpha_hats)/n_sim
        return rejectn_rate

    def beta(self, alpha):
        # TODO: replace this with binary search.
        ix = np.argmin((self.alphas-alpha)**2)
        return self.alphas[ix], self.betas[ix]


def main():
	dist1 = Multinom(3, 10)
	dist2 = Multinom(3, 10, ps=[.6,.2,.2])

	ab = AlphaBeta_1Sampl()
	ab.alpha_beta_tracer(dist1, dist2, get_p_val_chisq)
	ab2 = AlphaBeta_1Sampl()
	ab2.alpha_beta_tracer(dist1, dist2, get_pval_sim)
	plt.plot(ab.alphas, ab.beta)

