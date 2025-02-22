import numpy as np
from cvxopt import matrix, solvers
from scipy import optimize
from scipy.sparse import csr_matrix

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
        n, d = X.shape
        nlogL = lambda w: -y@np.log(sigmoid(X@w))-(1-y)@np.log(1-sigmoid(X@w))
        jac_nlogL = lambda w: -(X.T)@(y-sigmoid(X@w))
        hess_nlogL = lambda w: ((sigmoidp(X@w).reshape((n, 1))*X).T)@X
        w0 = np.zeros(d)
        res = optimize.minimize(nlogL, w0, method='Newton-CG', jac=jac_nlogL, hess=hess_nlogL)
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
        X : ndarray, shape (n,) or (n, d)
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
        X : ndarray, shape (n,) or (n, d)
            Features.

        Returns
        -------
        y : ndarray, shape (n,)
            Predicted labels.

        '''
        return self.kernel(X, self.training_points)@self.weights

class KernelLogisticRegression(object):
    '''
    Kernel Logistic Regression model.
    '''
    
    def __init__(self, kernel, l2reg=0.001):
        '''
        Kernel Logistic Regression init function.

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
        Kernel Logistic Regression fit function. Compute the weights based on
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
        y = 2*y-1
        K = self.kernel(X, X)
        loss = lambda w: np.mean(np.log(1+np.exp(-y*(K@w))))+self.l2reg*(w@K@w)
        jac_loss = lambda w: ((-sigmoid(-y*(K@w))*y)@K)/n+2.*self.l2reg*(K@w)
        hess_loss = lambda w: ((sigmoidp(y*(K@w))*(y**2)*K)@K)/n+2.*self.l2reg*K
        w0 = np.zeros(n)
        res = optimize.minimize(loss, w0, method='Newton-CG', jac=jac_loss, hess=hess_loss)
        print(res.message)
        self.training_points = X
        self.weights = res.x
    
    def predict(self, X):
        '''
        Kernel Logistic Regression predict function. Evaluate the model on the
        provided points.

        Parameters
        ----------
        X : ndarray, shape (n,) or (n, d)
            Features.

        Returns
        -------
        y : ndarray, shape (n,)
            Predicted labels.

        '''
        return sigmoid(self.kernel(X, self.training_points)@self.weights)

class KernelSVM(object):
    '''
    Kernel SVM model.
    '''
    
    def __init__(self, kernel, l2reg=0.001):
        '''
        Kernel SVM init function.

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
        Kernel SVM fit function. Compute the weights based on training points
        and labels using the provided kernel function.

        Parameters
        ----------
        X : ndarray, shape (n,) or (n, d)
            Features.
        y : ndarray, shape (n,)
            Labels.

        Returns
        -------
        None.

        '''
        n = X.shape[0]
        y = 2*y-1
        K = self.kernel(X, X)
        P = matrix(K, tc='d')
        q = matrix(-y, tc='d')
        G = matrix(np.concatenate((-np.diag(y), np.diag(y)), axis=0), tc='d')
        h = matrix(np.block([np.zeros(n), np.ones(n)/(2.*n*self.l2reg)]), tc='d')
        sol = solvers.qp(P, q, G, h)
        self.training_points = X
        self.weights = np.ravel(sol['x'])
        
    def predict(self, X):
        '''
        Kernel SVM predict function. Evaluate the model on the provided points.

        Parameters
        ----------
        X : ndarray, shape (n,) or (n, d)
            Features.

        Returns
        -------
        y : ndarray, shape (n,)
            Predicted labels.

        '''
        return self.kernel(X, self.training_points)@self.weights
