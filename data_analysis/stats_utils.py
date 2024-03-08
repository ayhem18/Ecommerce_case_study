"""
This script contains utility functions that are used across the different scripts in this case study
"""

import numpy as np
from typing import Sequence, Union, List
from scipy.stats import norm
from tqdm import tqdm
from statsmodels.distributions.empirical_distribution import ECDF  as smard_ecdf
from scipy.stats import kstest

def remove_outliers(x: Sequence):
    # this function simply removes any element that lies outside of the range  (q1 - 0.5 * IQR, q3 + 0.5 * IQR)
    q1, q3 = np.quantile(x, [0.25, 0.75])   
    iqr = q3 - q1
    return sorted([value for value in x if q1 - 0.5 * iqr <= value <= q1 + 0.5 * iqr])


def test_sample_normal_distribution(sample: Sequence[Union[int, float]], random_states: List[int] = None):
    if random_states is None: 
        random_states = [18, 31, 69, 71, 531]
    
    if isinstance(random_states, int):
        random_states = [random_states] 

    sample_size = len(sample)
    
    if sample_size <= 30:
        raise ValueError(f"The power of the 'Kolmogorov-Smirnov Test' is might be limited without a large sample of at least 30 elements. Found: {sample_size} elements") 

    sample_mean, sample_std = np.mean(sample), np.std(sample)
    
    p_values = []
    for rs in tqdm(random_states, desc='running normality test for random different random states'):
        # create the sample with the same std and mean as the sample
        norm_sample = norm.rvs(loc=sample_mean, 
                               scale=sample_std,
                               size=sample_size, 
                               random_state=rs)
        norm_cdf = norm.cdf(norm_sample, loc=sample, scale=sample_std)
        p_values.append(kstest(rvs=sample, cdf=norm_cdf).pvalue)

    return p_values


