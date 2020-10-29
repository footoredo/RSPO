import numpy as np
import torch
import os
import pathlib


def plot_statistics(statistics, keyword):
    import seaborn as sns
    import pandas as pd
    import matplotlib.pyplot as plt

    iters = []
    values = []
    names = []

    agents = list(statistics.keys())

    for agent in agents:
        s = statistics[agent][keyword]
        for it, v in s:
            iters.append(it)
            values.append(v)
            names.append(agent)

    df = pd.DataFrame(dict(iteration=iters, value=values, agent=names))
    fig, ax = plt.subplots()
    sns.lineplot(x="iteration", y="value", hue="agent", data=df, ax=ax)
    ax.set_title(keyword)
    plt.tight_layout()
    plt.show()


def mkdir(path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    return path


def mkdir2(path1, path2):
    return mkdir(os.path.join(path1, path2))


def ts(ar):
    return torch.from_numpy(np.array(ar, dtype=np.float32)).float()


def acquire_all_locks(locks):
    for lock in locks:
        lock.wait()
        lock.clear()


def release_all_locks(locks):
    for lock in locks:
        lock.set()


class Identity:
    '''
    A convenience class which simply implements __call__
    as the identity function
    '''
    def __call__(self, x, *args, **kwargs):
        return x

    def reset(self):
        pass


class RunningStat(object):
    '''
    Keeps track of first and second moments (mean and variance)
    of a streaming time series.
     Taken from https://github.com/joschu/modular_rl
     Math in http://www.johndcook.com/blog/standard_deviation/
    '''
    def __init__(self, shape):
        self._n = 0
        self._M = np.zeros(shape)
        self._S = np.zeros(shape)

    def push(self, x):
        x = np.asarray(x)
        assert x.shape == self._M.shape
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            oldM = self._M.copy()
            self._M[...] = oldM + (x - oldM) / self._n
            self._S[...] = self._S + (x - oldM) * (x - self._M)

    @property
    def n(self):
        return self._n

    @property
    def mean(self):
        return self._M

    @property
    def var(self):
        return self._S / (self._n - 1) if self._n > 1 else np.square(self._M)

    @property
    def std(self):
        return np.sqrt(self.var)

    @property
    def shape(self):
        return self._M.shape


class RewardFilter:
    """
    "Incorrect" reward normalization [copied from OAI code]
    Incorrect in the sense that we
    1. update return
    2. divide reward by std(return) *without* subtracting and adding back mean
    """

    def __init__(self, prev_filter, shape, gamma, clip=None):
        assert shape is not None
        self.gamma = gamma
        self.prev_filter = prev_filter
        self.rs = RunningStat(shape)
        self.ret = np.zeros(shape)
        self.clip = clip

    def __call__(self, x, **kwargs):
        x = self.prev_filter(x, **kwargs)
        self.ret = self.ret * self.gamma + x
        self.rs.push(self.ret)
        x = x / (self.rs.std + 1e-8)
        if self.clip:
            x = np.clip(x, -self.clip, self.clip)
        return x

    def reset(self):
        self.ret = np.zeros_like(self.ret)
        self.prev_filter.reset()