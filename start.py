import numpy as np
import pandas as pd
from datetime import datetime

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

# %% 

#########
# TO DO #
#########

Yte_predicted = pd.DataFrame(np.random.randint(0, 2, 3000), columns=["Bound"], index=Xte.index)

# %% SAVE PREDICTED VALUES

now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
Yte_predicted.to_csv("res/Yte_predicted_"+now+".csv")
