import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from a2c_ppo_acktr.multi_agent.utils import tsne
import joblib
import networkx as nx


def hashing(fg):
    h = 0
    for x in fg:
        h = h * 5 + x
    return h


def distance(fg1, fg2):
    return np.not_equal(fg1, fg2).sum()


def main():
    G = nx.DiGraph()
    fgs = []
    rs = []
    ps = set()
    cnt = 0
    all_data = []
    for r in range(1, 10):
        last = -1
        last_data = joblib.load("data/policy-fingerprints/fixed-init.run-{}.update-{}.data".format(r, 32))
        if last_data[0][0][0] == 2:
            continue
        for u in range(1, 33):
            _data = joblib.load("data/policy-fingerprints/fixed-init.run-{}.update-{}.data".format(r, u))
            for fg, e, m in _data:
                all_data.append((fg, r, u, e, m))
                if e == 0 and m == 0:
                    fgs.append(fg)
                    rs.append("run-{}".format(r))
                    h = hashing(fg)
                    cnt += 1
                    if last != -1 and last != h:
                        # print(last, h)
                        G.add_edge(last, h)
                    last = h
                    if h not in ps:
                        ps.add(h)
    # joblib.dump(all_data, "data/policy-fingerprints/all.data")
    print(len(ps), cnt)
    nx.draw_shell(G, node_size=30)
    plt.show()
    # tsne(fgs, rs, metric=distance)


if __name__ == "__main__":
    main()
