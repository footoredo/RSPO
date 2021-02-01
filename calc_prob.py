import numpy as np


n = 16
p = 0.5
r = 2. / 3.

sn = -int(-n * r)

C = np.zeros((n + 1, n + 1), dtype=np.int)

C[0, 0] = 1
for i in range(1, n + 1):
    C[i, 0] = 1
    for j in range(1, i + 1):
        C[i, j] = C[i - 1, j - 1] + C[i - 1, j]


def calc(m):
    rp = np.power(p, m) * np.power(1 - p, n - m) * C[n, m]
    return rp


prob = 0.
for _m in range(sn, n + 1):
    prob += calc(_m)

print(prob)
