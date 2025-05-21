import copy
from Hamil_search1 import *
from numpy.linalg import matrix_rank as rank


def getEq(eff):
    index = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 1, 2, 3)]
    res = [eff]
    for i in index:
        tmp = list(copy.deepcopy(eff))
        for j in i:
            tmp[j] *= -1
        res.append(tuple(tmp))
    return res


def equalEff(eff1, eff2):
    if eff1 == eff2:
        return True


class effs:
    def __init__(self, size, Xeff, Zeff) -> None:
        self.size = size
        self.xeff = Xeff
        self.zeff = Zeff

    def __eq__(self, value: object) -> bool:
        pass


def getSpace(n, H):
    eigenvalues, eigenvectors = LA.eigh(H)
    index = np.absolute(eigenvalues) < 1e-6
    eigenvectors = eigenvectors[:, index]
    return eigenvectors


def siftFile(source, target):
    # print(getEq((1, -3, -1, 3)))
    xeffs = set()
    zeffs = set()
    print("start")
    with open(source, "r") as f:
        with open(target, "w") as fw:
            content = f.readlines()
            for line in content:
                effs = line[9:-10]
                xeff, zeff = eval(effs)
                print(xeff, zeff)
                flagx = xeff in xeffs
                flagz = zeff in zeffs
                # print(xeffs, zeffs)
                # print(flagx, flagz)
                if flagx and flagz:
                    continue
                fw.write(f"{xeff}, {zeff}\n")
                if not flagx:
                    eqx = getEq(xeff)
                    for x in eqx:
                        xeffs.add(x)
                if not flagz:
                    eqz = getEq(zeff)
                    for z in eqz:
                        zeffs.add(z)


def EqSpan(H1, H2):
    C = np.concatenate((H1, H2), axis=1)
    r1 = rank(H1)
    r2 = rank(H2)
    r3 = rank(C)
    if r1 != r2:
        return False
    if r3 != r1:
        return False
    return True


if __name__ == "__main__":
    n = 8
    xeff = [1, 0] * (n // 2)
    zeff = [3, 0] * (n // 2)
    print(testEff(n, xeff, zeff))
    exit(0)
    with open("sift.txt", "r") as f:
        content = f.readlines()
        Pdict = []
        count = 0
        n = 4
        for line in content:
            count += 1
            xeff, zeff = eval(line)
            print(xeff, zeff, count)
            H = getHamil(n, xeff, zeff)
            flag = True
            for key in Pdict:
                Hin = key[0]
                if EqSpan(H, Hin):
                    key[1].append((xeff, zeff))
                    flag = False
                    break
            if flag:
                Pdict.append((H, [(xeff, zeff)]))
    with open("sift1.txt", "w") as fw:
        for i in range(len(Pdict)):
            fw.write(f"Group {i}\n")
            for eff in Pdict[i][1]:
                fw.write(f"{eff[0]}, {eff[1]}\n")
            fw.write("\n")
