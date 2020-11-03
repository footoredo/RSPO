import joblib
import numpy as np
import os


def load(path):
    evals, evecs = joblib.load(path)
    return evals, evecs


if __name__ == "__main__":
    load_dir = "./results/loaded-dice-simple-eigen/2020-11-02T09:31:28.657293/agent_0"
    path_1 = os.path.join(load_dir, "update-1", "eigen.data")
    path_2 = os.path.join(load_dir, "update-2", "eigen.data")

    evals1, evecs1 = load(path_1)
    evals2, evecs2 = load(path_2)

    n = evals1.shape[0]
    m = evals2.shape[0]

    for i in range(n):
        evec = evecs1[i]
        evec /= np.linalg.norm(evec)
        max_overlap = 0.
        for j in range(m):
            overlap = np.abs((evec.T @ evecs2[i]) / np.linalg.norm(evecs2[i]))
            max_overlap = max(overlap, max_overlap)
        print(i, evals1[i], max_overlap, np.linalg.norm(evec))
