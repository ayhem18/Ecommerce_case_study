"""
This script contains utility functions that are used across the different scripts in this case study
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Sequence, Union
from scipy.stats import norm, mode
from statsmodels.distributions.empirical_distribution import ECDF  as smart_ecdf
from scipy.stats import ks_1samp
from functools import partial
from bisect import bisect_left

def remove_outliers(x: Sequence):
    # this function simply removes any element that lies outside of the range  (q1 - 0.5 * IQR, q3 + 0.5 * IQR)
    q1, q3 = np.quantile(x, [0.25, 0.75])   
    iqr = q3 - q1
    return sorted([value for value in x if q1 - 0.5 * iqr <= value <= q1 + 0.5 * iqr])


def test_sample_normal_distribution(sample: Sequence[Union[int, float]], 
                                    display: bool = False,
                                    random_state: int = 60
                                    ):
    if isinstance(sample, pd.Series):
        sample = sample.values

    # make sure to sort the sample 
    sample = np.sort(sample)

    sample_size = len(sample)    
    if sample_size <= 30:
        raise ValueError(f"The power of the 'Kolmogorov-Smirnov Test' is might be limited without a large sample of at least 30 elements. Found: {sample_size} elements") 

    sample_mean, sample_std = np.mean(sample), np.std(sample)
    p_values = []

    # the code is based on the documentation of the 'scipy.kstest' function
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kstest.html
    
    normal_cdf_callable = partial(norm.cdf, loc=sample_mean, scale=sample_std)
    p_values.append(ks_1samp(x=sample, cdf=normal_cdf_callable).pvalue)

    if display:
        # create a normal sample
        norm_sample = norm.rvs(loc=sample_mean, 
                            scale=sample_std,
                            size=sample_size, 
                            random_state=random_state)

        sample_hf, norm_hf = np.max(np.histogram(sample)[0]), np.max(np.histogram(norm_sample)[0])
        hf = max(sample_hf, norm_hf)

        xticks_min = min(np.min(sample), np.min(norm_sample))
        xticks_max = max(np.max(sample), np.max(norm_sample))

        # we will display both the density and cumulativel distributions to better understand how the sample 
        # compares to the normal distribution (also for debugging purposes !!)
        fig = plt.figure(figsize=(20, 8))

        # the first plot will be the density function
        fig.add_subplot(1, 3, 1)
        plt.hist(sample, bins=20, density=False, label='sample', color='g') # 'g' for green
        plt.hist(norm_sample, bins=20, density=False, label='normal', color='r') # 'r' for red
        plt.xlabel('value')
        plt.ylabel('probability')
        plt.xticks(ticks=np.linspace(xticks_min, xticks_max, 26), rotation=90)
        # plt.yticks(np.linspace(0, 1, 26))
        plt.yticks([int(x) for x in np.linspace(1, hf, 26)])
        plt.legend()
        plt.title('density distribution')

        # create the emperical cdf for the sample 
        sample_cdf = smart_ecdf(sample)(sample) # the first call creates the emperical cdf object, the second returns the actual values
        normal_cdf = normal_cdf_callable(sample)

        fig.add_subplot(1, 3, 2)
        plt.plot(sample, sample_cdf, label='sample', color='g') # 'g' for green
        plt.plot(sample, normal_cdf, label='normal', color='r') # 'r' for red
        plt.xlabel('value')
        plt.ylabel('cumulative probability: P(X <= t)')
        plt.xticks(ticks=np.linspace(np.min(sample), np.max(sample), 26), rotation=90)
        plt.yticks(np.linspace(0, 1, 11))
        plt.legend()
        plt.title('cumulative distribution')


    return p_values


