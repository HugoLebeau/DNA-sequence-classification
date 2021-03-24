import numpy as np
from scipy.special import expit

sigmoid = expit
sigmoidp = lambda u: sigmoid(u)*sigmoid(-u)

def printStats(predicted, true, verbose=True):
    ''' Precision, recall, accuracy. '''
    precision = np.sum(predicted & true)/np.sum(predicted) # TP / (TP + FP)
    recall = np.sum(predicted & true)/np.sum(true) # TP / (TP + FN)
    accuracy = (np.sum(predicted & true)+np.sum((1-predicted) & (1-true)))/len(true) # (TP + TN) / (P + N)
    f1 = 2 * precision * recall / (precision + recall)
    if verbose:
        print("Precision\tRecall\tAccuracy\n{:.1%}\t\t{:.1%}\t{:.1%}".format(precision, recall, accuracy))
    else:
        return {"precision": precision, "recall": recall, "accuracy": accuracy, "f1-score": f1}
