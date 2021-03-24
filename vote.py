import pandas as pd
from datetime import datetime

files = ["res/Yte_predicted_KernelSVM_2021-03-23_22-47-23.csv", # KernelSVM (9,1)-mismatch
         "res/Yte_predicted_KernelSVM_2021-03-24_07-17-22.csv", # KernelSVM (9,2)-mismatch
         "res/Yte_predicted_KernelLogReg_2021-03-23_22-09-40.csv", #KernelLogReg (9,0)-mismatch
         "res/Yte_predicted_KernelSVM_2021-03-23_23-07-52.csv", #KernelSVM (10,1)-mismatch
         "res/Yte_predicted_KernelSVM_2021-03-23_23-13-47.csv"] #KernelSVM (11,1)-mismatch

dfs = [pd.read_csv(file, index_col=0) for file in files]
vote = (pd.concat(dfs, axis=1).sum(axis=1)/len(files)).round().astype(int)
vote.name = "Bound"

now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
name = "res/Yte_predicted_vote_"+now+".csv"
vote.to_csv(name)
print("Results saved in {}.".format(name))
