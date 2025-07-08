from Hamil_search1 import *

n = 5
L0 = ket2Vec(
    n,
    [
        "-00000",
        "10010",
        "01001",
        "01010",
        "11011",
        "-00110",
        "-11000",
        "11101",
        "-00011",
        "11110",
        "01111",
        "-10001",
        "-01100",
        "-10111",
        "00101",
        "-10100",
    ],
)

L1 = ket2Vec(
    n,
    [
        "11111",
        "01101",
        "-10110",
        "-01011",
        "-10101",
        "-00100",
        "11001",
        "00111",
        "-00010",
        "11100",
        "-00001",
        "-10000",
        "-01110",
        "-10011",
        "-01000",
        "-11010",
    ],
)


def isScalar(v1, v2):
    left = np.dot(v1, v2) ** 2
    right = np.dot(v1, v1) * np.dot(v2, v2)
    return np.abs(left - right) < 1e-6


def testH_codeword(H, codewords):
    Lambda = None
    for cw in codewords:
        res = H @ cw
        if not isScalar(res, cw):
            return False
        for i in range(res.shape[0]):
            if np.abs(res[i]) > 1e-6:
                break
        lamb = res[i] / cw[i]
        if not Lambda:
            Lambda = lamb
        else:
            if np.abs(lamb - Lambda) > 1e-6:
                return False
    return True


stabs = ["X0*Z1*Z2*X3", "X1*Z2*Z3*X4", "X0*X2*Z3*Z4", "Z0*X1"]
print(isScalar(np.array([1, 2]), np.array([2, 4.1])))
