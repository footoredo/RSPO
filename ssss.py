import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
fmri = sns.load_dataset("fmri")
sns.set()
mpl.rcParams['font.family'] = "monospace"
mpl.rcParams['text.color'] = "C1C1CD"
mpl.rcParams['patch.linewidth'] = 0.
mpl.rcParams['axes.facecolor'] = "F5F8F8"
mpl.rcParams['axes.edgecolor'] = "C1C1CD"
mpl.rcParams['xtick.color'] = "C1C1CD"
mpl.rcParams['ytick.color'] = "C1C1CD"
mpl.rcParams['grid.color'] = "E5E7EB"
sns.relplot(x="timepoint", y="signal", kind="line", data=fmri)
plt.show()
