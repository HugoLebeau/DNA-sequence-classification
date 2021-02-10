import numpy as np
import pandas as pd
from datetime import datetime

from functions import sigmoid, printStats, logistic_regression

np.random.seed(14159)

# %% LOAD DATA

# Training data
Xtr0 = pd.read_csv("data/Xtr0.csv", index_col="Id")
Xtr1 = pd.read_csv("data/Xtr1.csv", index_col="Id")
Xtr2 = pd.read_csv("data/Xtr2.csv", index_col="Id")
Xtr = pd.concat([Xtr0, Xtr1, Xtr2], axis=0)

Xtr0_mat100 = pd.read_csv("data/Xtr0_mat100.csv", sep=' ', header=None)
Xtr1_mat100 = pd.read_csv("data/Xtr1_mat100.csv", sep=' ', header=None)
Xtr2_mat100 = pd.read_csv("data/Xtr2_mat100.csv", sep=' ', header=None)
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

Xte0_mat100 = pd.read_csv("data/Xte0_mat100.csv", sep=' ', header=None)
Xte1_mat100 = pd.read_csv("data/Xte1_mat100.csv", sep=' ', header=None)
Xte2_mat100 = pd.read_csv("data/Xte2_mat100.csv", sep=' ', header=None)
Xte_mat100 = pd.concat([Xte0_mat100, Xte1_mat100, Xte2_mat100], axis=0).reset_index(drop=True)

# %% Train/eval split

prop_eval = 10/100 # proportion of the training set dedicated to evaluation
id_eval = np.random.choice(Xtr.index, np.int(Xtr.shape[0]*prop_eval), replace=False)
id_eval = np.isin(np.arange(Xtr.shape[0]), id_eval)
id_train = ~id_eval

# %%

Yte_predicted = pd.DataFrame(index=Xte.index)

# Logisitic regression
w_train = logistic_regression(Xtr_mat100[id_train].values, Ytr["Bound"][id_train].values)
predicted = np.where(sigmoid(np.dot(Xtr_mat100[id_eval], w_train)) > 0.5, 1, 0)
print("LOGISTIC REGRESSION")
printStats(predicted, Ytr["Bound"][id_eval].values)

w = logistic_regression(Xtr_mat100.values, Ytr["Bound"].values)
Yte_predicted["LogReg"] = np.where(sigmoid(np.dot(Xte_mat100, w)) > 0.5, 1, 0)

# %% SAVE PREDICTED VALUES

now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
for c in Yte_predicted.columns:
    Yte_predicted[[c]].rename(columns={c: "Bound"}).to_csv("res/Yte_predicted_"+c+"_"+now+".csv")
