import numpy as np
from scipy.spatial.distance import cdist

class RBF_kernel(object):
    def __init__(self, sigma=1.):
        self.sigma = sigma
        self.name = "RBF kernel"
    def __call__(self, X1, X2):
        pairwise_dists = cdist(X1, X2, "sqeuclidean")
        return np.exp(-pairwise_dists/(2*self.sigma**2))

class polynomial_kernel(object):
    def __init__(self, d, c=0.):
        self.d = d
        self.c = c
        if c < 0.:
            print("Parameter c = {} in polynomial kernel should be >= 0.".format(c))
        self.name = "Polynomial kernel of degree {}".format(d)
    def __call__(self, X1, X2):
        return (X1@(X2.T)+self.c)**self.d

class linear_kernel(polynomial_kernel):
    def __init__(self):
        super(linear_kernel, self).__init__(1, 0.)
        self.name = "Linear kernel"
