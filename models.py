import numpy as np
from scipy import optimize

from functions import sigmoid, sigmoidp

class LogisticRegression(object):
    '''
    Logistic Regression model class.
    '''
    
    def __init__(self):
        '''
        Logistic Regression init function.

        Returns
        -------
        None.

        '''
        self.training_points = None
        self.weights = None
    
    def fit(self, X, y):
        '''
        Logistic Regression fit function. Compute the weights based on
        training points and labels.

        Parameters
        ----------
        X : ndarray, shape (n, d)
            Features.
        y : ndarray, shape (n,)
            Labels.

        Returns
        -------
        None.

        '''
        n, p = X.shape
        nlogL = lambda w: -y@np.log(sigmoid(X@w))-(1-y)@np.log(1-sigmoid(X@w))
        grad_nlogL = lambda w: -(X.T)@(y-sigmoid(X@w))
        hess_nlogL = lambda w: ((sigmoidp(X@w).reshape((n, 1))*X).T)@X
        w0 = np.zeros(p)
        res = optimize.minimize(nlogL, w0, method='Newton-CG', jac=grad_nlogL, hess=hess_nlogL)
        if not res.success:
            print(res.message)
        self.training_points = X
        self.weights = res.x
    
    def predict(self, X):
        '''
        Logistic Regression predict function. Evaluate the model on the
        provided points.

        Parameters
        ----------
        X : ndarray, shape (n, d)
            Features.

        Returns
        -------
        y : ndarray, shape (n,)
            Predicted labels.

        '''
        return sigmoid(X@self.weights)

class KernelRidgeRegression(object):
    '''
    Kernel Ridge Regression model class.
    '''
    
    def __init__(self, kernel, l2reg=0.001):
        '''
        Kernel Ridge Regression init function.

        Parameters
        ----------
        kernel : callable(X1, X2) -> float
            Kernel function returning cross-kernel matrix between X1 and X2
            vectors.
        l2reg : float
            L2 Regularization parameter.

        Returns
        -------
        None.

        '''
        self.kernel = kernel
        self.l2reg = l2reg
        self.training_points = None
        self.weights = None

    def fit(self, X, y):
        '''
        Kernel Ridge Regression fit function. Compute the weights based on
        training points and labels using the provided kernel function.

        Parameters
        ----------
        X : ndarray, shape (n, d)
            Features.
        y : ndarray, shape (n,)
            Labels.

        Returns
        -------
        None.

        '''
        n = X.shape[0]
        K = self.kernel(X, X)
        weights = np.linalg.solve(K+self.l2reg*n*np.eye(n), y)
        self.training_points = X
        self.weights = weights

    def predict(self, X):
        '''
        Kernel Ridge Regression predict function. Evaluate the model on the
        provided points.

        Parameters
        ----------
        X : ndarray, shape (n, d)
            Features.

        Returns
        -------
        y : ndarray, shape (n,)
            Predicted labels.

        '''
        return self.kernel(X, self.training_points).dot(self.weights)
