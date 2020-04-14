## https://github.com/ryu577/scipy/pull/1/commits/9630185a7dac681497cb4a83958eb335f5938aca
def binom_test_v2(x, n=None, p=0.5, alternative='two-sided'):
    """
    Perform a test that the probability of success is p.
    This is an exact, two-sided test of the null hypothesis
    that the probability of success in a Bernoulli experiment
    is `p`.
    Parameters
    ----------
    x : int or array_like
        The number of successes, or if x has length 2, it is the
        number of successes and the number of failures.
    n : int
        The number of trials.  This is ignored if x gives both the
        number of successes and failures.
    p : float, optional
        The hypothesized probability of success.  ``0 <= p <= 1``. The
        default value is ``p = 0.5``.
    alternative : {'two-sided', 'greater', 'less'}, optional
        Indicates the alternative hypothesis. The default value is
        'two-sided'.
    Returns
    -------
    p-value : float
        The p-value of the hypothesis test.
    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Binomial_test
    Examples
    --------
    >>> from scipy import stats
    A car manufacturer claims that no more than 10% of their cars are unsafe.
    15 cars are inspected for safety, 3 were found to be unsafe. Test the
    manufacturer's claim:
    >>> stats.binom_test(3, n=15, p=0.1, alternative='greater')
    0.18406106910639114
    The null hypothesis cannot be rejected at the 5% level of significance
    because the returned p-value is greater than the critical value of 5%.
    """
    #x = atleast_1d(x).astype(np.int_)
    # if len(x) == 2:
    #     n = x[1] + x[0]
    #     x = x[0]
    # elif len(x) == 1:
    #     x = x[0]
    #     if n is None or n < x:
    #         raise ValueError("n must be >= x")
    #     n = np.int_(n)
    # else:
    #     raise ValueError("Incorrect length for x.")
    n = np.int_(n)
    if (p > 1.0) or (p < 0.0):
        raise ValueError("p must be in range [0,1]")

    if alternative not in ('two-sided', 'less', 'greater'):
        raise ValueError("alternative not recognized\n"
                         "should be 'two-sided', 'less' or 'greater'")

    if alternative == 'less':
        pval = binom.cdf(x, n, p)
        return pval

    if alternative == 'greater':
        pval = binom.sf(x-1, n, p)
        return pval

    # if alternative was neither 'less' nor 'greater', then it's 'two-sided'
    d = binom.pmf(x, n, p)
    rerr = 1 + 1e-7
    a_fn = lambda x1:binom.pmf(x1,n,p)
    if x == p * n:
        # special case as shortcut, would also be handled by `else` below
        pval = 1.
    elif x < p * n:
        #i = np.arange(np.ceil(p * n), n+1)
        #y = np.sum(binom.pmf(i, n, p) <= d*rerr, axis=0)        
        y = n-binary_search(a_fn,d*rerr,np.ceil(p * n),n)+1
        pval = (binom.cdf(x, n, p) +
                binom.sf(n - y, n, p))
    else:
        #i = np.arange(np.floor(p*n) + 1)
        #y = np.sum(binom.pmf(i, n, p) <= d*rerr, axis=0)
        y = binary_search(a_fn,d*rerr,0,np.floor(p*n) + 1,True)+1
        pval = (binom.cdf(y-1, n, p) +
                binom.sf(x-1, n, p))

    return min(1.0, pval)

  
def binary_search(a, d, lo, hi, asc_order=False):
    while lo < hi:
        mid = (lo+hi)//2
        midval = a(mid)
        if midval < d:
            if asc_order:
                lo = mid+1
            else:
                hi = mid-1
        elif midval > d:
            if asc_order:
                hi = mid-1
            else:
                lo = mid+1
        else:
            return mid
    if a(lo)<=d:
        return lo
    else:
        return lo-(asc_order-0.5)*2

from scipy.stats import binom_test
result=df
result["p_value"] = result.apply(lambda row : binom_test(row.Events_B,row.Events_A + row.Events_B,row.Uptime_B/(row.Uptime_A +row.Uptime_B),alternative='greater'), axis=1)
result["p_value_two_sided"] = result.apply(lambda row : binom_test(row.Events_B,row.Events_A + row.Events_B,row.Uptime_B/(row.Uptime_A +row.Uptime_B)), axis=1 )
