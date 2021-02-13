import numpy as np
from scipy.spatial.distance import cdist


def RBF_kernel(X1, X2, sigma=0.2):
    """
    Computes the RBF kernel between two sets of vectors
    """
    pairwise_dists = cdist(X1, X2, "sqeuclidean")
    return np.exp(-pairwise_dists / (2 * sigma ** 2))


class KernelRidgeRegression(object):
    """
    Kernel Ridge Regression model class.
    """

    def __init__(self, kernel, l2reg=0.001):
        """
        Kernel Ridge Regression init function.

        Parameters
        ----------
        kernel : callable(X1, X2) -> float
            Kernel function returning cross-kernel matrix between X1 and X2 vectors.
        l2reg : float
            L2 Regularization parameter.
        """

        self.kernel = kernel
        self.l2reg = l2reg
        self.training_points = None
        self.weights = None

    def fit(self, X, y):
        """
        Kernel Ridge Regression Fit function. Computes the weights based on training points and labels using the provided kernel function.

        Parameters
        ----------
        X : ndarray, shape (n, d)
            Features.
        y : ndarray, shape (n,)
            Labels.
        """
        K = self.kernel(X1=X, X2=X)
        weights = np.linalg.inv(K + self.l2reg * np.identity(K.shape[0])) @ y
        self.training_points = X
        self.weights = weights

    def predict(self, X):
        """
        Kernel Ridge Regression Predict function. Evaluates the kernel model on the provided points.

        Parameters
        ----------
        X : ndarray, shape (n, d)
            Features.
        Returns
        -------
        y : ndarray, shape (n,)
            Predicted labels.
        """
        return self.kernel(X, self.training_points).dot(self.weights)
