from Hamil_search import *
from itertools import product
from simulation import *
import time
COMMUTE = 0
MULTI = 1

def Cons2Str(constraint, PauliSet):
    ind = constraint.consIndex
    type = constraint.type
    eff = constraint.eff
    if type == COMMUTE:
        return f'{eff} commute with {PauliSet[ind[0]]}'
    elif type == MULTI:
        return f'{eff} product of {PauliSet[ind[0]]} and {PauliSet[ind[1]]}'
    return ""

class Cons:
    def __init__(self, consIndex, type, eff=1) -> None:
        self.consIndex = consIndex
        self.type = type
        self.eff= eff

    def __str__(self) -> str:
        return f'{self.eff}*{self.term}'

    def __repr__(self):
        return str(self)

def searchLogical(n,k,H):
    P = getProjector(n,H)
    phyCanStr = phyOpCandiate(n)
    phyCan = [pauliExpr2Mat(n, i) for i in phyCanStr]
    '''
    ConstraintSet, the first op is free so is []
    rules for each op is a list of CONS)
    '''
    '''
        order 
    
    '''
    ConstraintSet = []
    index = [(i,j) for i in range(k-1) for j in range(i+1, k)]
    PauliSet = [f'X{i}' for i in range(k)] + [f'Z{i}' for i in range(k)]
    PauliSet += [f'X{a[0]}X{a[1]}' for a in index] + [f'Z{a[0]}Z{a[1]}' for a in index]
    # add X0~X_{k-1}
    for i in range(k):
        rule = []
        for j in range(i):
            rule.append(Cons([j], COMMUTE, 1))
        ConstraintSet.append(rule)
    # Z0~Z_{n-1}
    for i in range(k):
        rule = []
        for j in range(k):
            if i!=j:
                rule.append(Cons([j], COMMUTE, 1))
            else:
                rule.append(Cons([j], COMMUTE, -1))
        for j in range(i):
            rule.append(Cons([k+j], COMMUTE, 1))   
        ConstraintSet.append(rule)  
    # add X0X1 ...
    for ind in index:
        a,b = ind        
        rule=[Cons([a,b], MULTI, 1)]
        ConstraintSet.append(rule)

    # add Z0Z1 ...
    for ind in index:
        a,b = ind        
        rule=[Cons([a+k,b+k], MULTI, 1)]
        ConstraintSet.append(rule)

    for i in range(len(ConstraintSet)):
        cs = ConstraintSet[i]
        tmp = [Cons2Str(c, PauliSet) for c in cs]
        print(f'{PauliSet[i]}: {", ".join(tmp)}')

    indexStack = [0]
    print(f"conSetLen: {len(ConstraintSet)}, candiateLen: {len(phyCan)}")

    while len(indexStack)>0:
        print(indexStack)
        if len(indexStack)>=len(ConstraintSet):
            break
        curLog = len(indexStack)-1
        curPhyInd = indexStack[curLog]
        if curPhyInd >= len(phyCan):
            indexStack.pop(-1)
            indexStack[-1] += 1
            continue
        phyOp = phyCan[curPhyInd]
        flag = True
        for cons in ConstraintSet[curLog]:
            relatedOp = [phyCan[i] for i in cons.consIndex]
            if not testConstraint(P, phyOp, cons, relatedOp):
                flag = False
                break
        if flag:
            if len(indexStack) == len(ConstraintSet):
                break
            else:
                indexStack.append(0)
        else:
            indexStack[-1] += 1

    return PauliSet, [phyCanStr[i] for i in indexStack]



def testConstraint(P, phyOp, cons, relatedOp):
    type = cons.type
    eff = cons.eff
    if type == COMMUTE:
        return commuteOrNot(P@phyOp, P@relatedOp[0], eff)
    if type == MULTI:
        return checkSame(P@phyOp, P@relatedOp[0] @ P@ relatedOp[1])
    return False

def Pstr2P(n, Pstr):
    c = np.kron(ket2Vec(n, Pstr[0]), ket2Vec(n, Pstr[1])) / 2
    v = c.reshape((4**n, 1))
    return v @ dagger(v)

def dagger(v):
    return np.conj(v).T 

if __name__ =='__main__':
    
    blocksize = 4
    blockNum = 3
    n = blocksize * blockNum
    start = time.time()
    # Pstr1 = [['00', '11'], ['01', '-10']]
    # Pstr2 = [['00', '-11'], ['01', '10']]
    # Pstr3 = [['01', '10'], ['00', '-11']]
    # Pstr4 = [['01', '-10'], ['00', '11']]
    # Pstrs = [Pstr1, Pstr2, Pstr3, Pstr4]
    # Ps = sum([Pstr2P(2, p) for p in Pstrs])
    # P = Ps
    # for i in range(1, blockNum):
    #     P = np.kron(P, Ps)
    U = getU(blocksize, blockNum)
    P = U @ U.conj().T
    end = time.time()
    print(f"get P use {end-start}s")
    # P = P / 2**4
    Q = np.identity(2**n) - P
    print(MEqual(P@P, P))
    g = 3
    Xeff = (1,0) * (n//2)
    Zeff = (g,0) * (n//2)
    Hpen = getHamil(n, Xeff, Zeff)
    end = time.time()
    print(f'get Hpen use {end-start}s')
    HpenInverse = np.linalg.pinv(Hpen)
    end = time.time()
    print(f'get HpenInverse use {end-start}s')
    T = Hpen @ HpenInverse @ Hpen
    # print(MEqual(T, Hpen))
    lambdaPen = 16
    Q0 = HpenInverse @ Hpen
    P0 = np.identity(Q0.shape[0]) - Q0
    def crossTerm(i):
        return [(1, f'Z{4*i+1}*X{4*i+7}+Z{4*i+3}*X{4*i+7}')]
    Henc_block = getHencBlock(blockNum, lambdaPen=16, crossTerm=crossTerm)
    Htar_block = getHtarBlock(n, blockNum)
    Henc = blocks2Mat(n, Henc_block)
    end = time.time()
    print(f'get Henc use {end-start}s')
    Htar = blocks2Mat(n//2, Htar_block)
    end =time.time()
    print(f"get Htar use {end-start}s")
    Heff_2nd_order = P0@Henc@P0 - (P0@Henc@Q0@HpenInverse@Q0@Henc@P0 / lambdaPen)
    print(checkSame(P @ Heff_2nd_order @ P, U@Htar@U.conj().T))
    def A2LA(A):
        return P @ A @ Q @ HpenInverse @ Q @ A @ P
    A = pauliExpr2Mat(n, 'X1*Z4+X2*Z4')
    la = A2LA(A)
    print(LA.norm(la))
    def getEff(n, pauliList):   
        pmap = {'I': I, 'X':X, 'Y':Y, 'Z':Z}
        Ms = [pmap[k]/2 for k in pauliList]
        res = 0
        def getIndex(m, j):
            return m//2, m%2, j//2, j%2
        
        Vs = [np.kron(ket2Vec(n//4, Pstr[0]), ket2Vec(n//4, Pstr[1])) / 2 for Pstr in Pstrs]

        def vkron(m,j):
            return np.kron(Vs[m], Vs[j]).reshape((1, 2**n))
        for mj in range(16):
            for mj_ in range(16):
                m = mj // 4
                j = mj % 4
                m_ = mj_ // 4
                j_ = mj_ % 4
                m1, m0, j1, j0 = getIndex(m, j)
                m1_, m0_, j1_, j0_ = getIndex(m_, j_)
                eff = Ms[0][m1_, m1] * Ms[1][m0_, m0] * Ms[2][j1_, j1] * Ms[3][j0_, j0]
                prod = vkron(m,j) @ la @ dagger(vkron(m_, j_))
                # if eff!=0:
                #     print(mj, mj_, eff, prod)
                #     print(m1, m0, j1, j0, m1_, m0_, j1_, j0_)
                res += eff * prod
        return res
    plist = ('I', 'X', 'Y', 'Z')
    res = getEff(n, 'IIZI')
    print(res)
    # exit(0)
    for i in plist:
        for j in plist:
            for k in plist:
                for l in plist:
                    s = i+j+k+l
                    result = getEff(n, s)
                    if abs(result) > 1e-4:
                        print(s, result)
    # n = 6
    # k = 3
    # Xeff, Zeff = (0, -1, -2, -1, 0, 1), (2, 0, -2, 0, 2, 1)
    # H = getHamil(n,Xeff,Zeff)
    # list, res = searchLogical(n, k, H)
    # print(list)
    # print(res)