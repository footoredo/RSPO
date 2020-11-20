import joblib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def main():
    xs = []
    ys = []
    hs = []
    axs = []
    ahs = []
    for i in range(1, 33):
        cnt = joblib.load("grad-{}.cnt.obj".format(i))
        xs.append(i)
        ys.append(cnt[0])
        hs.append("reach-0")
        xs.append(i)
        ys.append(cnt[1])
        hs.append("reach-1")

        for _ in range(cnt[0]):
            axs.append(i)
            ahs.append("reach-0")
        for _ in range(cnt[1]):
            axs.append(i)
            ahs.append("reach-1")

    df = pd.DataFrame(data=dict(x=xs, y=ys, h=hs))
    adf = pd.DataFrame(data=dict(x=axs, h=ahs))
    # sns.lineplot(x="x", y="y", hue="h", data=df)
    sns.displot(data=adf, x="x", hue="h", kind="kde", multiple="fill", clip=(0, None), palette="ch:rot=-.25,hue=1,light=.75")
    plt.show()


if __name__ == '__main__':
    main()
