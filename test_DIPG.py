import math
import numpy as np


def main():
    path = "/tmp/collect_trajectories/2021-05-19T15:51:18.939085#e468/2021-05-19T15:51:21.009587#b727/0/trajectories.obj.npy"
    a = np.load(path)

    a /= np.std(a)

    n, m = a.shape
    k = 5
    g = np.random.randn(m, k) / math.sqrt(m) / math.sqrt(k)

    def ker(x, y):
        d = np.matmul(x, g) - np.matmul(y, g)
        # print(d)
        # print(np.std(d), np.square(d).sum())
        # print(d)
        return math.pow(math.sqrt(2 * math.pi), -k) * math.exp(-np.square(d).sum() / 2)

    print(ker(a[0], a[1]))


if __name__ == "__main__":
    main()
