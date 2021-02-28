import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from functions import printStats
from kernels import RBF_kernel, linear_kernel
from models import LogisticRegression, KernelRidgeRegression, KernelLogisticRegression, KernelSVM

np.random.seed(14159)

method = "KernelSVM" # "LogReg" / "KernelRidge" / "KernelLogReg" / "KernelSVM"
kernel = RBF_kernel(sigma=1e-1)
# kernel = linear_kernel()

print("Chosen method: {}.\n".format(method))

# %% LOAD DATA

# Training data
Xtr0 = pd.read_csv("data/Xtr0.csv", index_col="Id")
Xtr1 = pd.read_csv("data/Xtr1.csv", index_col="Id")
Xtr2 = pd.read_csv("data/Xtr2.csv", index_col="Id")
Xtr = pd.concat([Xtr0, Xtr1, Xtr2], axis=0)

Xtr0_mat100 = pd.read_csv("data/Xtr0_mat100.csv", sep=" ", header=None)
Xtr1_mat100 = pd.read_csv("data/Xtr1_mat100.csv", sep=" ", header=None)
Xtr2_mat100 = pd.read_csv("data/Xtr2_mat100.csv", sep=" ", header=None)
Xtr_mat100 = pd.concat([Xtr0_mat100, Xtr1_mat100, Xtr2_mat100], axis=0).reset_index(drop=True)

Ytr0 = pd.read_csv("data/Ytr0.csv", index_col="Id")
Ytr1 = pd.read_csv("data/Ytr1.csv", index_col="Id")
Ytr2 = pd.read_csv("data/Ytr2.csv", index_col="Id")
Ytr = pd.concat([Ytr0, Ytr1, Ytr2], axis=0)

# Test data
Xte0 = pd.read_csv("data/Xte0.csv", index_col="Id")
Xte1 = pd.read_csv("data/Xte1.csv", index_col="Id")
Xte2 = pd.read_csv("data/Xte2.csv", index_col="Id")
Xte = pd.concat([Xte0, Xte1, Xte2], axis=0)

Xte0_mat100 = pd.read_csv("data/Xte0_mat100.csv", sep=" ", header=None)
Xte1_mat100 = pd.read_csv("data/Xte1_mat100.csv", sep=" ", header=None)
Xte2_mat100 = pd.read_csv("data/Xte2_mat100.csv", sep=" ", header=None)
Xte_mat100 = pd.concat([Xte0_mat100, Xte1_mat100, Xte2_mat100], axis=0).reset_index(drop=True)

# %% TRAIN/EVAL SPLIT

prop_eval = 10/100  # proportion of the training set dedicated to evaluation
id_eval = np.random.choice(Xtr.index, np.int(Xtr.shape[0]*prop_eval), replace=False)
id_eval = np.isin(np.arange(Xtr.shape[0]), id_eval)
id_train = ~id_eval

# %% PREDICTION

Yte_predicted = pd.DataFrame(index=Xte.index)

if method == "LogReg":
    model = LogisticRegression()
    threshold = 0.5
elif method == "KernelRidge":
    model = KernelRidgeRegression(kernel, l2reg=1e-5)
    threshold = 0.5
elif method == "KernelLogReg":
    model = KernelLogisticRegression(kernel, l2reg=1e-5)
    threshold = 0.5
elif method == "KernelSVM":
    model = KernelSVM(kernel, l2reg=1e-5)
    threshold = 0.

print("Training...")
model.fit(Xtr_mat100[id_train].values, Ytr["Bound"][id_train].values)
print("Done.\n")
predicted = model.predict(Xtr_mat100[id_eval].values)
predicted_labels = np.where(predicted > threshold, 1, 0)

predicted0 = predicted[Ytr["Bound"][id_eval].values == 0]
predicted1 = predicted[Ytr["Bound"][id_eval].values == 1]
plt.hist([predicted0, predicted1], stacked=True, edgecolor='black', bins=20, label=[0, 1])
plt.axvline(threshold, ls='--', color='black')
plt.title("Histogram of predicted values")
plt.legend(title="True label")
plt.show()

printStats(predicted_labels, Ytr["Bound"][id_eval].values)

print("\nTraining...")
model.fit(Xtr_mat100.values, Ytr["Bound"].values)
print("Done.")
Yte_predicted[method] = np.where(model.predict(Xte_mat100) > threshold, 1, 0)

# %% SAVE PREDICTED VALUES

now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
for c in Yte_predicted.columns:
    Yte_predicted[[c]].rename(columns={c: "Bound"}).to_csv("res/Yte_predicted_"+c+"_"+now+".csv")
