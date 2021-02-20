import numpy as np
from scipy.spatial.distance import cdist

def RBF_kernel(X1, X2, sigma=0.2):
    '''
    Compute the RBF kernel between two sets of vectors.
    '''
    pairwise_dists = cdist(X1, X2, "sqeuclidean")
    return np.exp(-pairwise_dists/(2*sigma**2))
