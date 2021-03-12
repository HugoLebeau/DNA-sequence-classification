import numpy as np
from scipy.sparse import csr_matrix
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

class spectrum_kernel(object):
    def __init__(self, k):
        self.k = k
        self.name = "{}-spectrum kernel".format(k)
    def __call__(self, X1, X2):
        vocabulary, Phi = {}, []
        first = True
        for X in [X1, X2]:
            '''
            Count the number of occurences of each word of length k.
            Every new word in X1 is saved.
            Only words already seen in X1 are considered in X2.
            '''
            indptr, indices, data = [0], [], []
            for seq in X:
                for i in range(len(seq)-self.k+1):
                    if first or (seq[i:i+self.k] in vocabulary):
                        index = vocabulary.setdefault(seq[i:i+self.k], len(vocabulary))
                        indices.append(index)
                        data.append(1)
                indptr.append(len(indices))
            Phi.append(csr_matrix((data, indices, indptr), dtype=int)) # build a CSR matrix
            first = False
        return Phi[0].dot(Phi[1].transpose()).toarray()
