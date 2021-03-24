import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from functions import printStats
from kernels import RBF_kernel, linear_kernel, spectrum_kernel, mismatch_kernel
from models import LogisticRegression, KernelRidgeRegression, KernelLogisticRegression, KernelSVM

parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, default="KernelSVM", metavar="METHOD")
parser.add_argument('--l2reg', type=float, default=1e-5, metavar="L2REG")
parser.add_argument('--k', type=int, default=10, metavar="K")
parser.add_argument('--m', type=int, default=0, metavar="M")
parser.add_argument('--threshold', type=str, default=None, metavar="T")
args = parser.parse_args()

# %% INITIALISATION

np.random.seed(14159)

# kernel = RBF_kernel(sigma=1e-1)
# kernel = linear_kernel()
# kernel = spectrum_kernel(args.k)
kernel = mismatch_kernel(args.k, args.m)

if args.method == "LogReg":
    model = LogisticRegression()
    threshold = 0.5
elif args.method == "KernelRidge":
    model = KernelRidgeRegression(kernel, l2reg=args.l2reg)
    threshold = 0.5
elif args.method == "KernelLogReg":
    model = KernelLogisticRegression(kernel, l2reg=args.l2reg)
    threshold = 0.5
elif args.method == "KernelSVM":
    model = KernelSVM(kernel, l2reg=args.l2reg)
    threshold = 0.
else:
    raise NotImplementedError

mat100 = False

optimize_threshold = (args.threshold == "optimize")

print("Chosen method: {}.".format(args.method))
print("Chosen kernel: {}.\n".format(kernel.name))

# %% LOAD DATA

# Training data
if mat100:
    Xtr0 = pd.read_csv("data/Xtr0_mat100.csv", sep=" ", header=None)
    Xtr1 = pd.read_csv("data/Xtr1_mat100.csv", sep=" ", header=None)
    Xtr2 = pd.read_csv("data/Xtr2_mat100.csv", sep=" ", header=None)
    Xtr = pd.concat([Xtr0, Xtr1, Xtr2], axis=0).reset_index(drop=True).values
else:
    Xtr0 = pd.read_csv("data/Xtr0.csv", index_col="Id")
    Xtr1 = pd.read_csv("data/Xtr1.csv", index_col="Id")
    Xtr2 = pd.read_csv("data/Xtr2.csv", index_col="Id")
    Xtr = pd.concat([Xtr0, Xtr1, Xtr2], axis=0)["seq"].values

Ytr0 = pd.read_csv("data/Ytr0.csv", index_col="Id")
Ytr1 = pd.read_csv("data/Ytr1.csv", index_col="Id")
Ytr2 = pd.read_csv("data/Ytr2.csv", index_col="Id")
Ytr = pd.concat([Ytr0, Ytr1, Ytr2], axis=0)["Bound"].values

# Test data
if mat100:
    Xte0 = pd.read_csv("data/Xte0_mat100.csv", sep=" ", header=None)
    Xte1 = pd.read_csv("data/Xte1_mat100.csv", sep=" ", header=None)
    Xte2 = pd.read_csv("data/Xte2_mat100.csv", sep=" ", header=None)
    Xte = pd.concat([Xte0, Xte1, Xte2], axis=0).reset_index(drop=True).values
else:
    Xte0 = pd.read_csv("data/Xte0.csv", index_col="Id")
    Xte1 = pd.read_csv("data/Xte1.csv", index_col="Id")
    Xte2 = pd.read_csv("data/Xte2.csv", index_col="Id")
    Xte = pd.concat([Xte0, Xte1, Xte2], axis=0)["seq"].values

# %% TRAIN/EVAL SPLIT

prop_eval = 10/100  # proportion of the training set dedicated to evaluation
id_eval = np.random.choice(Xtr.shape[0], int(Xtr.shape[0]*prop_eval), replace=False)
id_eval = np.isin(np.arange(Xtr.shape[0]), id_eval)
id_train = ~id_eval

# %% PREDICTION

Yte_predicted = pd.DataFrame(index=range(Xte.shape[0]))
Yte_predicted.index.name = "Id"

print("Training...")
model.fit(Xtr[id_train], Ytr[id_train])
print("Done.\n")
predicted = model.predict(Xtr[id_eval])

if optimize_threshold:
    thresholds = np.linspace(threshold-0.4, threshold+0.4, 17)
    benchmarks = []
    for thr in thresholds:
        benchmarks.append(printStats(np.where(predicted > thr, 1, 0), Ytr[id_eval], verbose=False))
    
    optimal_metric = {
        "accuracy": np.max([metric["accuracy"] for metric in benchmarks]),
        "recall": np.max([metric["recall"] for metric in benchmarks]),
        "precision": np.max([metric["precision"] for metric in benchmarks]),
        "f1-score": np.max([metric["f1-score"] for metric in benchmarks]),
        }
    
    optimal_threshold = {
        "accuracy": thresholds[np.argmax([metric["accuracy"] for metric in benchmarks])],
        "recall": thresholds[np.argmax([metric["recall"] for metric in benchmarks])],
        "precision": thresholds[np.argmax([metric["precision"] for metric in benchmarks])],
        "f1-score": thresholds[np.argmax([metric["f1-score"] for metric in benchmarks])],
        }
    
    metric_considered = "accuracy"
    best_threshold = optimal_threshold[metric_considered]
    best_score = optimal_metric[metric_considered]
    print(f"Best threshold at {round(best_threshold, 4)} with {metric_considered} {round(best_score, 4)}.")
else:
    best_threshold = threshold

predicted0 = predicted[Ytr[id_eval] == 0]
predicted1 = predicted[Ytr[id_eval] == 1]
plt.hist([predicted0, predicted1], stacked=True, edgecolor='black', bins=20, label=[0, 1])
plt.axvline(best_threshold, ls='--', color='black')
plt.title("Histogram of predicted values (eval)")
plt.legend(title="True label")
plt.show()

predicted_labels = np.where(predicted > best_threshold, 1, 0)

printStats(predicted_labels, Ytr[id_eval])

print("\nTraining...")
model.fit(Xtr, Ytr)
print("Done.\n")

Yte_predicted[args.method] = np.where(model.predict(Xte) > best_threshold, 1, 0)

# %% SAVE PREDICTED VALUES

now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
for c in Yte_predicted.columns:
    name = "res/Yte_predicted_"+c+"_"+now+".csv"
    Yte_predicted[[c]].rename(columns={c: "Bound"}).to_csv(name)
    print("Results saved in {}.".format(name))
