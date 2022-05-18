#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import numpy as np
from sklearn.neighbors import KernelDensity


def random_sampling(
    values, size, method="histogram", n_bins=100, bandwidth=0.2, rtol=1e-4
):
    """
    Perform a random sampling of the rows of the input array based on
    the distribution of the columns values.

    :param values: Input array of values N x M, where N >> M
    :param size: Number of indices (rows) to be extracted from the original array

    :returns: Indices of samples randomly selected from the PDF
    """
    if size == values.shape[0]:
        return np.where(np.all(~np.isnan(values), axis=1))[0]
    else:
        if method == "pdf":
            kde_skl = KernelDensity(bandwidth=bandwidth, rtol=rtol)
            kde_skl.fit(values)
            probabilities = np.exp(kde_skl.score_samples(values))
            probabilities /= probabilities.sum()
        else:
            probabilities = np.zeros(values.shape[0])
            for ind in range(values.shape[1]):
                vals = values[:, ind]
                nnan = ~np.isnan(vals)
                pop, bins = np.histogram(vals[nnan], n_bins)
                ind = np.digitize(vals[nnan], bins)
                ind[ind > n_bins] = n_bins
                probabilities[nnan] += 1.0 / (pop[ind - 1] + 1)

    probabilities[np.any(np.isnan(values), axis=1)] = 0
    probabilities /= probabilities.sum()

    np.random.seed = 0
    return np.random.choice(
        np.arange(values.shape[0]), replace=False, p=probabilities, size=size
    )
