import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from functions import printStats
from kernels import RBF_kernel, linear_kernel
from models import LogisticRegression, KernelRidgeRegression, KernelLogisticRegression, KernelSVM

np.random.seed(14159)

method = "KernelSVM" # "LogReg" / "KernelRidge" / "KernelLogReg" / "KernelSVM"
kernel = linear_kernel()

print("choosen method :" , method)

# %% LOAD DATA

# Training data
print("\nloading the data...")
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
print("done.\n")

# %% TRAIN/EVAL SPLIT
print("splitting...")
prop_eval = 10/100  # proportion of the training set dedicated to evaluation
id_eval = np.random.choice(Xtr.index, np.int(Xtr.shape[0]*prop_eval), replace=False)
id_eval = np.isin(np.arange(Xtr.shape[0]), id_eval)
id_train = ~id_eval
print("done.\n")

# %%

Yte_predicted = pd.DataFrame(index=Xte.index)

if method == "LogReg":
    model = LogisticRegression()
elif method == "KernelRidge":
    model = KernelRidgeRegression(kernel, l2reg=1e-5)
elif method == "KernelLogReg":
    model = KernelLogisticRegression(kernel, l2reg=1e-5)
elif method == "KernelSVM":
    model = KernelSVM(kernel, l2reg=1e-5)
print("training...")
model.fit(Xtr_mat100[id_train].values, Ytr["Bound"][id_train].values)
print("done.\n")
predicted = model.predict(Xtr_mat100[id_eval])
print("printing...")
plt.hist(predicted, edgecolor='black', bins=20)
plt.axvline(0.5, ls='--', color='black')
plt.title("Histogram of predicted values")
plt.show()

predicted_labels = np.where(predicted > 0.5, 1, 0)

printStats(predicted_labels, Ytr["Bound"][id_eval].values)

model.fit(Xtr_mat100.values, Ytr["Bound"].values)
Yte_predicted[method] = np.where(model.predict(Xte_mat100) > 0.5, 1, 0)

# %% SAVE PREDICTED VALUES

now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
for c in Yte_predicted.columns:
    Yte_predicted[[c]].rename(columns={c: "Bound"}).to_csv("res/Yte_predicted_"+c+"_"+now+".csv")
