import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator, PercentFormatter

# Read data
res = pd.read_csv("res/res.csv", sep=';')

# Pre-processing
float_cols = ["l2reg", "Precision", "Recall", "Accuracy"]
res[float_cols] = res[float_cols].stack().str.replace(',', '.').unstack().astype(float)

# Compute F-score
res["F-score"] = 2./(1./res["Precision"]+1./res["Recall"])

# Sort values by F-score
res.sort_values(by="F-score", ascending=False, inplace=True)

# Plot
plt.plot(res["F-score"].values, label='F-score')
plt.plot(res["Accuracy"].values, label='Accuracy')
plt.grid(ls=':')
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.gca().yaxis.set_major_formatter(PercentFormatter())
plt.legend()
plt.show()

stats = ["F-score", "Accuracy"]
methods = ["KernelLogReg", "KernelSVM"]
for stat in stats:
    fig, ax = plt.subplots(1, len(stats))
    norm = Normalize(res[stat].min(), res[stat].max())
    for k, met in enumerate(methods):
        score = res[res["Method"] == met][["k", "m", stat]].set_index("k").pivot(columns="m")
        ax[k].imshow(score, cmap='bwr', norm=norm)
        xticks = range(score.shape[1])
        yticks = range(score.shape[0])
        ax[k].set_xticks(xticks)
        ax[k].set_xticklabels(score.columns.get_level_values(1))
        ax[k].set_yticks(yticks)
        ax[k].set_yticklabels(score.index)
        for i in xticks:
            for j in yticks:
                ax[k].text(i, j, score.iloc[j, i].round(1), ha='center', va='center', fontsize='large')
        ax[k].set_title(met)
        ax[k].set_xlabel("m")
        ax[k].set_ylabel("k")
    fig.suptitle(stat)
    plt.show()
