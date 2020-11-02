import numpy as np


def lanczos_method(A):
    n = A.shape[0]

    v = np.random.randn(n, 1)
    v /= np.linalg.norm(v)
    vs = [v]
    wp = A @ v
    a = wp.T @ v
    w = wp - a * v

    eps = 1e-3
    m = 20
    T = np.zeros((m, m))
    T[0, 0] = a

    for j in range(1, m):
        b = np.linalg.norm(w)
        T[j - 1, j] = b
        T[j, j - 1] = b
        lv = v
        if b > eps or b < -eps:
            v = w / b
        else:
            print("resample!")
            v = np.random.randn(n, 1)
            for vv in v:
                v -= (v.T @ vv) * vv
            v /= np.linalg.norm(v)
        wp = A @ v
        a = wp.T @ v
        w = wp - a * v - b * lv
        vs.append(v)
        # print(a, b)

        T[j, j] = a

    evals, evecs = np.linalg.eigh(T)

    V = torch.cat(vs, 1).detach().numpy()
    print(V.shape)
    qs = []
    for i in range(m):
        z = V @ evecs[i]
        z = z / np.linalg.norm(z)
        for q in qs:
            z = z - (z.T @ q) * q
        qs.append(z)

        # print(z.shape, z)
        print(evals[i], hv(fg, torch.tensor(z).float()).numpy() / z)