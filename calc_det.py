import numpy as np

# path = "sim_matrix.obj"
# path = "sync-results/hopper/512/0/sim_matrix.obj"
# path = "sync-results/hopper-512/find-all-auto/2021-05-26T02:38:11.976909#d0ad/sim_matrix.obj"
# path = "sync-results/hopper/dvd/0/sim_matrix.obj"
# path = "sync-results/hopper-512/dipg-find-all-auto/2021-05-28T01:56:34.486328#4985/sim_matrix.obj"

# path = "sync-results/walker2d/0-512-new/sim_matrix.obj"
# path = "sync-results/walker2d/find-all-auto/2021-05-25T18:51:03.779702#a237/sim_matrix.obj"
# path = "sync-results/walker2d/dvd/0/sim_matrix.obj"
# path = "sync-results/walker2d-512/dipg-find-all-auto/2021-05-28T03:23:09.476673#1538/sim_matrix.obj"

# path = "sync-results/half-cheetah/512/0/sim_matrix.obj"
# path = "sync-results/half-cheetah/dvd/0/sim_matrix.obj"
# path = "sync-results/half-cheetah-512/find-all-auto/2021-05-26T05:16:59.802762#e337/sim_matrix.obj"
# path = "sync-results/half-cheetah/find-all-auto/2021-05-24T06:43:37.103623#f2da/sim_matrix.obj"
# path = "sync-results/half-cheetah-512/dipg-find-all-auto/2021-05-28T00:25:30.563858#366d/sim_matrix.obj"

# path = "sync-results/stag-hunt-gw/vpg-5m/sim_matrix.obj"
# path = "sync-results/stag-hunt-gw/find-all/2021-05-06T16:11:28.028312#d782/sim_matrix.obj"
path = "sync-results/stag-hunt-gw-2m/dipg-find-all-auto/2021-05-28T03:01:47.532745#e008/sim_matrix.obj"

# path = "sync-results/escalation-gw/vpg-5m/sim_matrix.obj"
# path = "sync-results/escalation-gw/find-all-final-3/2021-05-26T09:57:52.976256#bcbc/sim_matrix.obj"
# path = "sync-results/escalation-gw-2m/dipg-find-all-auto/2021-05-28T05:36:57.309800#af09/sim_matrix.obj"

sim_matrix = np.load(path, allow_pickle=True)
print(sim_matrix)
l = 12
n = sim_matrix.shape[0]
# sim_matrix[1, 2] = sim_matrix[2, 1] = 3.
ker_matrix = np.zeros_like(sim_matrix)
for i in range(n):
    for j in range(n):
        ker_matrix[i, j] = np.exp(-sim_matrix[i, j] / (l * l))
print(ker_matrix)
print(np.linalg.det(ker_matrix))