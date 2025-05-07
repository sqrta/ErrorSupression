from Hamil_search import *
from itertools import product

stabs = [
    "Z0*Z3",
    "X6*X7",
    "X1*X2",
    "Z5*Z8",
    "X0*X1",
    "-1*X3*X4",
    "Z1*Z2",
    "-1*Z4*Z5",
    "Z3*Z4",
    "-1*Z6*Z7",
    "X4*X5",
    "-1*X7*X8",
]
surface = [
    "Z0*Z3",
    "X6*X7",
    "X1*X2",
    "Z5*Z8",
    "X0*X1*X3*X4",
    "Z1*Z2*Z4*Z5",
    "Z3*Z4*Z6*Z7",
    "X4*X5*X7*X8",
]

perfect = ["X0*Z1*Z2*X3", "X1*Z2*Z3*X4", "X0*X2*Z3*Z4", "Z0*X1*X3*Z4"]

candidate = []


def splitFour(stab, stabLen, outNum):
    terms = stab.split("*")
    result = []
    for comb in itertools.combinations(list(range(stabLen)), outNum):
        origin = list(range(stabLen))
        for index in comb:
            origin.remove(index)
        result.append(
            ("*".join([terms[i] for i in origin]), "*".join([terms[i] for i in comb]))
        )
    return result


previous = []
fourL = perfect
n = 5
print(fourL[0])

ss = [splitFour(s, 4, 2) for s in fourL]
candidate = list(product(*ss))
print(len(candidate))
count = 0
print(f"{len(candidate)} candidates")
step = 2
correct = []
wrong = []
for i in range(0, len(candidate), step):
    c = candidate[i]
    count += 1

    twoL = previous[:]
    twoCount = len(twoL)
    if twoCount > 0:
        twoL += [f"-{twoCount}*I0"]
    for j in range(step):
        c = candidate[i + j]
        eff = 1
        for ss in c:
            twoL.append(f"{eff}*" + ss[0])
            twoL.append(f"{-eff}*" + ss[1])
            eff *= 1
    # print(twoL)
    # exit(0)
    H = sum([pauliExpr2Mat(n, s) for s in twoL])
    config = {"target": 0, "distance": 2, "depth": 1, "thres": 0}
    # eigenvectors = getSpace(n, H, config)
    # print(count, eigenvectors.shape[1])
    result = testH(n, H, config)
    if result[0]:
        correct.append(twoL)
        print(f"{count} is correct")
        print(twoL)
    else:
        print(f"{count} is wrong")
        wrong.append(twoL)
        print(twoL)
with open("correctP.txt", "w") as f:
    for t in correct:
        f.write(str(t) + "\n")
with open("wrongP.txt", "w") as f:
    for t in wrong:
        f.write(str(t) + "\n")
# H = sum([pauliExpr2Mat(n, s) for s in perfect])
# config = {"target": 4, "distance": 2, "depth": 1, "thres": 0}
# result = testH(n, H, config)
# print(result)
