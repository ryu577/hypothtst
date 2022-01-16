import numpy as np
from collections import Counter


def get_pval(tst_stat=4, n=10, k=5, t_stat_fn=max, n_sim=10000):
	"""
	args:
		tst_stat: The test statistic calculated from the data.
		n: Number of packet drops
		k: Number of participants in the call
	"""
	ps = np.ones(k)/k
	p_val = 0
	for _ in range(n_sim):
		ys = np.random.choice(k, p=ps, size=n)
		ns = Counter(ys).values()
		ns = list(ns)
		sim_tst_stat = t_stat_fn(ns)
		if sim_tst_stat > tst_stat:
			p_val += 1/n_sim
	return p_val
