n = 100

f = [0.] * (n + 1)
for i in range(1, n + 1):
    f[i] = f[i - 1] + n / i

print(f[n])