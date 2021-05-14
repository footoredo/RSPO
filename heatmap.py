import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def plot(gather_count):
    sns.color_palette("light:b", as_cmap=True)
    ax = sns.heatmap(gather_count, vmax=8, vmin=0, cmap="Purples")
    plt.show()


def main():
    gather_count = [[0, 0, 0, 0, 0], [0, 0, 1, 0, 1], [0, 1, 2, 0, 4], [0, 0, 0, 7, 24], [0, 0, 5, 18, 4549]]
    gather_count = np.array(gather_count)
    gather_count = np.log(gather_count + 1)
    plot(gather_count)


if __name__ == "__main__":
    main()
