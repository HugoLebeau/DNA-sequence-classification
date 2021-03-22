import numpy as np
from tqdm import tqdm
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
            indptr, indices, data = [0], [], [] # these will be filled in order to build a CSR matrix
            for seq in tqdm(X):
                for i in range(len(seq)-self.k+1):
                    if first or (seq[i:i+self.k] in vocabulary):
                        index = vocabulary.setdefault(seq[i:i+self.k], len(vocabulary))
                        indices.append(index)
                        data.append(1)
                indptr.append(len(indices))
            shape = None if first else (X2.shape[0], Phi[0].shape[1])
            mat = csr_matrix((data, indices, indptr), dtype=int, shape=shape)
            mat.sum_duplicates()
            Phi.append(mat)
            first = False
        return Phi[0].dot(Phi[1].transpose()).toarray()

class mismatch_kernel(object):
    def __init__(self, k, m):
        self.k = k
        self.m = m
        self.name = "({}, {})-mismatch kernel".format(k, m)
    def __call__(self, X1, X2):
        
        def matching(seq, data, indices, vocabulary, alphabet, k, m, i=0, term='', mismatches=0):
            ''' Depth first search for k-mers matching with seq. '''
            if mismatches <= m:
                if i < k:
                    for char in alphabet:
                        if seq[i] == char:
                            matching(seq, data, indices, vocabulary, alphabet, k, m, i+1, term+char, mismatches)
                        else:
                            matching(seq, data, indices, vocabulary, alphabet, k, m, i+1, term+char, mismatches+1)
                if i == k:
                    index = vocabulary.setdefault(term, len(vocabulary))
                    indices.append(index)
                    data.append(1)
        
        alphabet = set(''.join(X1)).union(set(''.join(X2)))
        vocabulary, Phi = {}, []
        first = True
        for X in [X1, X2]:
            indptr, indices, data = [0], [], [] # these will be filled in order to build a CSR matrix
            for seq in tqdm(X):
                for i in range(len(seq)-self.k+1):
                    matching(seq[i:i+self.k], data, indices, vocabulary, alphabet, self.k, self.m, 0, '', 0)
                indptr.append(len(indices))
            shape = None if first else (X2.shape[0], Phi[0].shape[1])
            mat = csr_matrix((data, indices, indptr), dtype=int, shape=shape)
            mat.sum_duplicates()
            Phi.append(mat)
            first = False
        
        # return Phi[0].dot(Phi[1].transpose()).toarray()
        # dot product may cause segmentation fault on large matrices
        # implementation below is much slower, but does not crash
        out = np.zeros((Phi[0].shape[0], Phi[1].shape[0]), dtype=int)
        for i in tqdm(range(Phi[1].shape[0])):
            out[:, i] = Phi[0].dot(Phi[1].getrow(i).transpose()).toarray().ravel()
        return out
