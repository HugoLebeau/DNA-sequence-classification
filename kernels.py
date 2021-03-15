import numpy as np
from tqdm import tqdm
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist

from trie import Trie

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
            indptr, indices, data = [0], [], [] # these will be filled in order to build a CSR matrix
            for seq in tqdm(X):
                for i in range(len(seq)-self.k+1):
                    if first or (seq[i:i+self.k] in vocabulary):
                        index = vocabulary.setdefault(seq[i:i+self.k], len(vocabulary))
                        indices.append(index)
                        data.append(1)
                indptr.append(len(indices))
            shape = None if first else (X2.shape[0], Phi[0].get_shape()[1])
            Phi.append(csr_matrix((data, indices, indptr), dtype=int, shape=shape))
            first = False
        return Phi[0].dot(Phi[1].transpose()).toarray()

class mismatch_kernel(object):
    def __init__(self, k, m):
        self.k = k
        self.m = m
        self.name = "({}, {})-mismatch kernel".format(k, m)
    def __call__(self, X1, X2):
        vocabulary, Phi = {}, []
        first = True
        trie = Trie()
        for X in [X1, X2]:
            indptr, indices, data = [0], [], [] # these will be filled in order to build a CSR matrix
            for seq in tqdm(X):
                for i in range(len(seq)-self.k+1):
                    ls = [seq[i:i+self.k]] if first else trie.match(seq[i:i+self.k], self.m)
                    for s in ls:
                        index = vocabulary.setdefault(s, len(vocabulary))
                        indices.append(index)
                        data.append(1)
                indptr.append(len(indices))
            if first: # build Trie
                for key in vocabulary.keys():
                    trie.add_key(key)
            shape = None if first else (X2.shape[0], Phi[0].get_shape()[1])
            Phi.append(csr_matrix((data, indices, indptr), dtype=int, shape=shape))
            first = False
        return Phi[0].dot(Phi[1].transpose()).toarray()
