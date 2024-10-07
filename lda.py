import numpy as np

a = np.array([[4, 1], [2, 4], [2, 3], [3, 6], [4, 4]])
b = np.array([[9, 10], [6, 8], [9, 5], [8, 7], [10, 8]])

u1 = np.mean(a, axis=0)
u2 = np.mean(b, axis=0)


def mul(x, u):
    s = np.zeros((x.shape[1], x.shape[1]))
    for i in x:
        diff = (i - u).reshape(-1, 1)
        s += np.matmul(diff, diff.T)
    return s


s1 = mul(a, u1)
s2 = mul(b, u2)
sw = s1 + s2
sb = np.outer((u1 - u2), (u1 - u2).T)

e, v = np.linalg.eig((np.matmul(np.linalg.inv(sw), sb)))

s = np.argsort(e)[::-1]
e = e[s]
v = v[:, s]
w = v[:, 0]

a = np.dot(a, w)
b = np.dot(b, w)
print(a, b)