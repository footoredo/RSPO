import numpy as np

# path = "sim_matrix.obj"
# path = "sync-results/hopper/512/0/sim_matrix.obj"
# path = "sync-results/hopper-512/find-all-auto/2021-05-26T02:38:11.976909#d0ad/sim_matrix.obj"
# path = "sync-results/hopper/dvd/0/sim_matrix.obj"
# path = "sync-results/hopper-512/dipg-find-all-auto/2021-05-28T01:56:34.486328#4985/sim_matrix.obj"
# path = "sync-results/hopper/512/parallel/2021-08-08T11:31:41.228188#4350/sim_matrix.obj"

# path = "sync-results/walker2d/0-512-new/sim_matrix.obj"
# path = "sync-results/walker2d/find-all-auto/2021-05-25T18:51:03.779702#a237/sim_matrix.obj"
# path = "sync-results/walker2d/dvd/0/sim_matrix.obj"
# path = "sync-results/walker2d-512/dipg-find-all-auto/2021-05-28T03:23:09.476673#1538/sim_matrix.obj"
# path = "sync-results/walker2d/parallel/2021-08-08T12:51:19.644743#89e5/sim_matrix.obj"
# path = "sync-results/walker2d/parallel/2021-08-08T20:30:51.173656#b6f6/sim_matrix.obj"  # coef=0.5 0.35251131324406976
# path = "sync-results/walker2d/parallel/2021-08-08T22:00:47.199430#6e23/sim_matrix.obj"  # coef=0.5 0.6613482302822032
# path = "sync-results/walker2d/parallel/2021-08-09T09:22:20.977360#940a/sim_matrix.obj"  # coef=0.0 0.34935358372673114
# path = "sync-results/walker2d/parallel/2021-08-09T11:14:52.939445#c17e/sim_matrix.obj"  # coef=0.0 0.4150841542862816
# path = "sync-results/walker2d/parallel/2021-08-09T11:53:22.535535#e767/sim_matrix.obj"  # coef=0.0 0.1833098077305511
# path = "sync-results/walker2d/parallel/2021-08-09T14:04:16.804828#bd2d/sim_matrix.obj"  # coef=0.5 0.5210010740586771
# path = "sync-results/walker2d/parallel/2021-08-09T14:42:34.102494#6bad/sim_matrix.obj"  # coef=0.5 0.34346916904883784
path = "sync-results/walker2d/parallel/2021-08-09T15:20:46.655536#f710/sim_matrix.obj"  # coef=0.5 0.2237080188546646

# path = "sync-results/half-cheetah/512/0/sim_matrix.obj"
# path = "sync-results/half-cheetah/dvd/0/sim_matrix.obj"
# path = "sync-results/half-cheetah-512/find-all-auto/2021-05-26T05:16:59.802762#e337/sim_matrix.obj"
# path = "sync-results/half-cheetah/find-all-auto/2021-05-24T06:43:37.103623#f2da/sim_matrix.obj"
# path = "sync-results/half-cheetah-512/dipg-find-all-auto/2021-05-28T00:25:30.563858#366d/sim_matrix.obj"
# path = "sync-results/half-cheetah/512/parallel/2021-08-08T09:09:36.842470#a10d/sim_matrix.obj"
# path = "sync-results/half-cheetah-512/find-all-auto/2021-06-01T14:35:49.148740#fab2/sim_matrix.obj"

# path = "sync-results/stag-hunt-gw/vpg-5m/sim_matrix.obj"
# path = "sync-results/stag-hunt-gw/find-all/2021-05-06T16:11:28.028312#d782/sim_matrix.obj"
# path = "sync-results/stag-hunt-gw-2m/dipg-find-all-auto/2021-05-28T03:01:47.532745#e008/sim_matrix.obj"

# path = "sync-results/escalation-gw/vpg-5m/sim_matrix.obj"
# path = "sync-results/escalation-gw/find-all-final-3/2021-05-26T09:57:52.976256#bcbc/sim_matrix.obj"
# path = "sync-results/escalation-gw-2m/dipg-find-all-auto/2021-05-28T05:36:57.309800#af09/sim_matrix.obj"

sim_matrix = np.load(path, allow_pickle=True)
print(sim_matrix)
l = 30
n = sim_matrix.shape[0]
# sim_matrix[1, 2] = sim_matrix[2, 1] = 3.
ker_matrix = np.zeros_like(sim_matrix)
for i in range(n):
    for j in range(n):
        ker_matrix[i, j] = np.exp(-sim_matrix[i, j] / (l * l))
print(ker_matrix)
print(np.linalg.det(ker_matrix))