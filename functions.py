import numpy as np

sigmoid = lambda u: 1./(1.+np.exp(-u))
sigmoidp = lambda u: sigmoid(u)*(1.-sigmoid(u))

def printStats(predicted, true):
    ''' Precision, recall, accuracy '''
    precision = np.sum(predicted & true)/np.sum(predicted) # TP / (TP + FP)
    recall = np.sum(predicted & true)/np.sum(true) # TP / (TP + FN)
    accuracy = (np.sum(predicted & true)+np.sum((1-predicted) & (1-true)))/len(true) # (TP + TN) / (P + N)
    print("Precision\tRecall\tAccuracy\n{:.1%}\t\t{:.1%}\t{:.1%}".format(precision, recall, accuracy))
