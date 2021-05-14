import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import json

font = {'fontname': 'Inconsolata'}


import matplotlib
print(matplotlib.get_cachedir())

plt.rcParams["font.family"] = "monospace"

data = json.load(open("data.json"))
df = pd.DataFrame(data)
sns.catplot(x="i_alpha", y="reward", data=df)
plt.tight_layout()
# plt.xlabel("tetetetete12344", **font)
plt.show()
