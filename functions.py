import numpy as np
# import pandas as pd
from scipy import optimize

sigmoid = lambda u: 1./(1.+np.exp(-u))
sigmoidp = lambda u: sigmoid(u)*(1.-sigmoid(u))

def printStats(predicted, true):
    ''' Precision, recall, accuracy '''
    precision = np.sum(predicted & true)/np.sum(predicted) # TP / (TP + FP)
    recall = np.sum(predicted & true)/np.sum(true) # TP / (TP + FN)
    accuracy = (np.sum(predicted & true)+np.sum((1-predicted) & (1-true)))/len(true) # (TP + TN) / (P + N)
    print("Precision\tRecall\tAccuracy\n{:.1%}\t\t{:.1%}\t{:.1%}\n".format(precision, recall, accuracy))

def logistic_regression(X, y):
    '''
    Logistic regression y|X ~ B(s(Xw)). Compute w given X and y.

    Parameters
    ----------
    X : ndarray, shape (n, p)
        Features.
    y : ndarray, shape (n,)
        Labels.

    Returns
    -------
    ndarray, shape (p,)
        Predicted parameters.

    '''
    n, p = X.shape
    nlogL = lambda w: -np.dot(y, np.log(sigmoid(np.dot(X, w))))-np.dot(1-y, np.log(1-sigmoid(np.dot(X, w))))
    grad_nlogL = lambda w: -np.dot(X.T, y-sigmoid(np.dot(X, w)))
    hess_nlogL = lambda w: np.dot((sigmoidp(np.dot(X, w)).reshape((n, 1))*X).T, X)
    w0 = np.zeros(p)
    res = optimize.minimize(nlogL, w0, method='Newton-CG', jac=grad_nlogL, hess=hess_nlogL)
    if not res.success:
        print(res.message)
    return res.x
