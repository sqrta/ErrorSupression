import numpy as np
from numpy import kron
import functools
from scipy import linalg as LA
from itertools import product
import time
import random
import sys
import itertools
from functools import reduce


PAULI_MATRICES = np.array((((0, 1), (1, 0)), ((0, -1j), (1j, 0)), ((1, 0), (0, -1))))
X = np.array((((0, 1), (1, 0))))
Y = np.array(((0, -1j), (1j, 0)))
Z = np.array(((1, 0), (0, -1)))

I = np.identity(2)


def tensor(op_list):
    return functools.reduce(kron, op_list, 1)


def pauli2Mat(num_qubits, indexes, paulis):
    """
    pauli str to numpy matrices
    """
    op_list = [np.eye(2)] * num_qubits
    for index, pauli in zip(indexes, paulis):
        op_list[index] = pauli
    return tensor(op_list)


def getIsingHtarBlock(blockNum):
    Htar_block = []
    for i in range(blockNum):
        Htar_block.append((1, f"Z{2*i}*Z{2*i+1}"))
        Htar_block.append((1, f"Z{2*i}"))
        Htar_block.append((1, f"Z{2*i+1}"))
        Htar_block.append((1, f"X{2*i}"))
        Htar_block.append((1, f"X{2*i+1}"))
    for i in range(blockNum - 1):
        # B1.append((1, f'Z{2*i+1}*Z{2*i+2}'))
        # B2.append((1, f'Z{2*i+2}'))
        Htar_block.append((1, f"Z{2*i+1}*Z{2*i+2}"))
    return Htar_block


def pauliStr2mat(num_qubits, pstrings):
    indexes = []
    paulis = []
    pmap = {"I": I, "X": X, "Y": Y, "Z": Z}
    pauli = pstrings.split("*")
    eff = 1
    if pauli[0].isdigit():
        eff = eval(pauli[0])
        pauli.pop(0)
    for p in pauli:
        paulis.append(pmap[p[0].upper()])
        indexes.append(int(p[1:]))
    return eff * pauli2Mat(num_qubits, indexes, paulis)


def splitPaulis(e):
    pstr = e
    pauli = set(["I", "X", "Y", "Z"])

    effStr = ["1"]
    pStr = []
    for i in e.split("*"):
        if i[0] not in pauli:
            effStr.append(i)
        else:
            pStr.append((i[0], int(i[1:])))
    effstr = "*".join(effStr)

    eff = eval(effstr)
    return eff, pStr


def parsePauliTerm(e):
    pstr = e
    eff = 1
    pauli = set(["I", "X", "Y", "Z"])
    if e[0] not in pauli:
        effStr = []
        pStr = []
        for i in e.split("*"):
            if i[0] not in pauli:
                effStr.append(i)
            else:
                pStr.append(i)
        effstr = "*".join(effStr)
        pstr = "*".join(pStr)

        eff = eval(effstr)
    return eff, pstr


def pauliExpr2Mat(n, expr):
    """
    n: size
    pstring: e.g. X1*X2 + Z1*Z2
    """
    exp = expr.split("+")
    terms = []
    for e in exp:
        eff, pstr = parsePauliTerm(e)
        terms.append(PauliTerm(n, pstr, eff))
    H = sum([t.value() for t in terms])
    return H


def vec2Ket(vec):
    args = np.where(np.absolute(vec) > 1e-4)[0]
    res = [(vec[a], a) for a in args]

    return res


def ket2Str(n, kets):

    strings = [f"({a[0]:.7f})|{a[1]:0{n}b}>" for a in kets]
    return " + ".join(strings)


class PauliTerm:
    def __init__(self, n, term, eff=1) -> None:
        self.eff = eff
        self.term = term
        self.n = n
        # print(eff, term)

    def value(self):
        return self.eff * pauliStr2mat(self.n, self.term)

    def __str__(self) -> str:
        return f"{self.eff}*{self.term}"

    def __repr__(self):
        return str(self)


def ket2Vec(n, kets):
    """
    ket2Vec(n, ['1000', '-0111']
    """
    vec = np.zeros((2**n, 1))
    for ket in kets:
        sign = 1
        res = None
        if ket[0] == "-":
            sign = -1
            res = ket[1:]
        else:
            res = ket
        index = int(res, base=2)
        vec[index, 0] = sign
    return vec / (len(kets)) ** 0.5


def checkSame(P1, P2, thres=1e-4):
    nonZ = np.nonzero(np.absolute(P1 - P2) > 1e-4)
    if len(nonZ[0]) == 0:
        return True
    return False


def checkLinear(P1, P2):
    indexes = np.nonzero(np.absolute(P1) > 1e-4)
    if len(indexes[0]) == 0:
        return True
    i1 = (indexes[0][0], indexes[1][0])
    if np.absolute(P2[i1]) < 1e-5:
        return False
    r1 = P1[i1] / P2[i1]
    return checkSame(P1, r1 * P2)


def allSubset2Size(s, startSize, endSize):
    return reduce(
        lambda a, b: a + b, [findsubsets(s, i) for i in range(startSize, endSize + 1)]
    )


def findsubsets(s, n):
    return list(itertools.combinations(s, n))


def getErrorStr(n, distance):
    result = []
    indexSet = allSubset2Size(list(range(n)), 1, distance)
    for i in range(1, distance + 1):
        paulis = list(itertools.product(["X", "Y", "Z"], repeat=i))
        indexSet = findsubsets(list(range(n)), i)
        for index in indexSet:
            for p in paulis:
                result.append("*".join([f"{p[i]}{index[i]}" for i in range(len(p))]))
    return result


def testProjector(P, n, config):
    distance = config["distance"]
    errorStr = getErrorStr(n, distance)
    for es in errorStr:
        E = pauliStr2mat(n, es)
        res = P @ E @ P
        if not checkLinear(res, P):
            return False

    return True


def getP(vecs):
    return sum([vec @ vec.conj().T for vec in vecs])


def bindTerm(n, eff, name):
    X = []
    for i in range(n):
        if eff[i] != 0:
            tmp = f"{name}{i}*{name}{(i+1)%n}"
            if eff[i] == -1:
                tmp = "-" + tmp
            elif eff[i] != 1:
                tmp = f"{eff[i]}*" + tmp
            if eff[i] > 0 and i > 0:
                tmp = "+" + tmp
            X.append(tmp)
    X = "".join(X)
    return X


def XZeff2Str(n, Xeff, Zeff):
    print(Xeff)
    print(Zeff)
    X = bindTerm(n, Xeff, "X")
    Z = bindTerm(n, Zeff, "Z")

    return X, Z


def phyOpCandiate(n):
    index = [(i, j) for i in range(n - 1) for j in range(i, n)]
    res = [f"X{i}" for i in range(n)] + [f"Z{i}" for i in range(n)]
    for ind in index:
        a, b = ind
        res += [f"X{a}*X{b}", f"X{a}*Z{b}", f"Z{a}*X{b}", f"Z{a}*Z{b}"]
    return res


def printVecs(n, Xeff, Zeff):
    terms = [PauliTerm(n, f"X{i}*X{(i+1)%n}", Xeff[i]) for i in range(n)]
    terms += [PauliTerm(n, f"Z{i}*Z{(i+1)%n}", Zeff[i]) for i in range(n)]
    H = sum([t.value() for t in terms])
    eigenvalues, eigenvectors = LA.eigh(H)

    index = np.absolute(eigenvalues) < 1e-6
    eigenvectors = eigenvectors[:, index]

    for i in range(eigenvectors.shape[1]):
        vec = eigenvectors[:, i]
        print(vec.shape)
        print("-----------")
        print("-----------")
        # print(H @ vec.T)
        print(ket2Str(n, vec2Ket(vec)))


def getHamil(n, Xeff, Zeff):
    terms = [PauliTerm(n, f"X{i}*X{(i+1)%n}", Xeff[i]) for i in range(n)]
    terms += [PauliTerm(n, f"Z{i}*Z{(i+1)%n}", Zeff[i]) for i in range(n)]
    # print(terms)
    H = sum([t.value() for t in terms])
    return H


def getSpace(n, H, config):
    target = config["target"]
    eigenvalues, eigenvectors = np.linalg.eig(H)
    if target == "min":
        minEigen = min(eigenvalues)
        index = np.absolute(eigenvalues - minEigen) < 1e-6
    else:
        index = np.absolute(eigenvalues - int(target)) < 1e-6
    eigenvectors = eigenvectors[:, index]
    return eigenvectors


def testH(n, H, config):
    eigenvectors = getSpace(n, H, config)
    PList = []
    print(f"space size: {eigenvectors.shape[1]}")
    if eigenvectors.shape[1] < 1:
        return False, eigenvectors.shape[1]
    for i in range(eigenvectors.shape[1]):
        vec = eigenvectors[:, i]
        vec = vec.reshape(len(vec), 1)
        PList.append(vec)
    # PList = [ket2Vec(n, ['1000', '0111']), ket2Vec(n, ['0100', '1011']), ket2Vec(n, ['0010', '1101']), ket2Vec(n, ['0001', '1110'])]
    P = getP(PList)
    result = testProjector(P, n, config)
    # if result:
    #     print(eigenvectors.shape[1])
    return result, eigenvectors.shape[1]


def getProjector(n, H, config):

    eigenvectors = getSpace(n, H, config)
    PList = []
    # print(f"space size: {eigenvectors.shape[1]}")
    if eigenvectors.shape[1] < 4:
        return False, eigenvectors.shape[1]
    for i in range(eigenvectors.shape[1]):
        vec = eigenvectors[:, i]
        vec = vec.reshape(len(vec), 1)
        PList.append(vec)
    # PList = [ket2Vec(n, ['1000', '0111']), ket2Vec(n, ['0100', '1011']), ket2Vec(n, ['0010', '1101']), ket2Vec(n, ['0001', '1110'])]
    P = getP(PList)
    return P


def testLogicalOp(n, pauliStr, H, Pc, config):
    op = pauliExpr2Mat(n, pauliStr)
    P = getProjector(n, H, config)
    O = P @ op @ P
    return commuteOrNot(O, Pc)


def commuteOrNot(P1, P2, sign=1):
    """
    test P1@P2 - sign * P2@P1
    """
    M = P1 @ P2 - sign * P2 @ P1
    if LA.norm(M) < 1e-4:
        return True
    return False


def MEqual(M1, M2):
    if LA.norm(M1 - M2) < 1e-4:
        return True
    return False


def testEff(n, Xeff, Zeff, config):
    terms = [PauliTerm(n, f"X{i}*X{(i+1)%n}", Xeff[i]) for i in range(n)]
    terms += [PauliTerm(n, f"Z{i}*Z{(i+1)%n}", Zeff[i]) for i in range(n)]
    # print(terms)

    H = sum([t.value() for t in terms])
    return testH(n, H, config)


def searchHpen(n, config, path="result"):
    k = config["depth"]
    thres = config["thres"]
    with open(path, "w") as f:
        # if True:
        Xeff_can = list(product(range(-k, k + 1), repeat=n))
        Zeff_can = list(product(range(-k, k + 1), repeat=n))
        random.shuffle(Xeff_can)
        random.shuffle(Zeff_can)
        for Xeff in Xeff_can:
            if Xeff[0] < 0:
                continue
            print("xeff", Xeff)
            for Zeff in Zeff_can:
                if Zeff[0] < 0:
                    continue

                res, size = testEff(n, Xeff, Zeff, config)
                # print('zeff', Zeff, res, size)
                if res and size >= thres:
                    f.write(
                        (
                            f"succeed: {Xeff}, {Zeff}, size: {size}, {XZeff2Str(n, Xeff, Zeff)}\n"
                        )
                    )
                    # print((f"succeed: {Xeff}, {Zeff}, size: {size}"))


if __name__ == "__main__":
    n = 6
    depth = 3
    thres = 0

    if len(sys.argv) > 1:
        n = int(sys.argv[1])
    if len(sys.argv) > 2:
        depth = int(sys.argv[2])
    if len(sys.argv) > 3:
        thres = int(sys.argv[3])
    config = {"target": "min", "distance": 2, "depth": depth, "thres": thres}
    # c1 = ket2Vec(n, ['1000', '0111'])
    # P = c1 @ c1.conj().T
    # print(P)
    # PList = [ket2Vec(n, ['1000', '0111']), ket2Vec(n, ['0100', '1011']), ket2Vec(n, ['0010', '1101']), ket2Vec(n, ['0001', '1110'])]
    # P = getP(PList)
    # E = pauliStr2mat(n, f'X{0}*X1+Z0*Z2')
    # res = P @ E @ P
    # print(checkLinear(res, P))
    # exit(0)
    start = time.time()

    # Xeff = sum([[0,1] for i in range(n//2)], start = [])
    # Zeff = sum([[0,-1] for i in range(n//2)], start=[])
    # print(Xeff)
    # print(Zeff)
    # printVecs(n, Xeff, Zeff)
    # res = testEff(n, Xeff, Zeff)
    # print(res)
    # exit(0)
    # res = testEff(n, Xeff, Zeff)
    # print(res)
    searchHpen(
        n, config, f"result{n}_{depth}_{thres}_{config['distance']}_{config['target']}"
    )
    end = time.time()
    print(f"use {end-start}s")
