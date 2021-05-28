import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def plot(gather_count, filename):
    gather_count = np.log(gather_count + 1)
    sns.color_palette("light:b", as_cmap=True)
    ax = sns.heatmap(gather_count, vmax=8, vmin=0, cmap="Purples",
                     xticklabels=False, yticklabels=False, cbar=False,
                     square=True)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    [i.set_linewidth(2) for i in ax.spines.values()]
    plt.tight_layout()
    # plt.show()
    plt.savefig(filename)


def main():
    gather_count = [[0, 0, 0, 0, 0], [0, 0, 1, 0, 1], [0, 1, 2, 0, 4], [0, 0, 0, 7, 24], [0, 0, 5, 18, 4549]]
    gather_count = np.array(gather_count)
    gather_count = np.log(gather_count + 1)
    plot(gather_count)


if __name__ == "__main__":
    main()
