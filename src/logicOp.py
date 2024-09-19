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
    
def debugMsg(debug, *msg):
    if debug:
        debugMsg(debug, *msg)

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

def getLogicOp(getHtarBlock, blocksize, blockNum, Astring, debug=False):
    n = blocksize * blockNum
    start = time.time()
    Pstr1 = [['00', '11'], ['01', '-10']]
    Pstr2 = [['00', '-11'], ['01', '10']]
    Pstr3 = [['01', '10'], ['00', '-11']]
    Pstr4 = [['01', '-10'], ['00', '11']]
    Pstrs = [Pstr1, Pstr2, Pstr3, Pstr4]
    # Ps = sum([Pstr2P(2, p) for p in Pstrs])
    # P = Ps
    # for i in range(1, blockNum):
    #     P = np.kron(P, Ps)
    U = getU(blocksize, blockNum)
    P = U @ U.conj().T
    end = time.time()
    debugMsg(debug, f"get P use {end-start}s")
    # P = P / 2**4
    Q = np.identity(2**n) - P
    debugMsg(debug, MEqual(P@P, P))
    g = 3
    Xeff = (1,0) * (n//2)
    Zeff = (g,0) * (n//2)
    Hpen = getHamil(n, Xeff, Zeff)
    end = time.time()
    debugMsg(debug, f'get Hpen use {end-start}s')
    HpenInverse = np.linalg.pinv(Hpen)
    end = time.time()
    debugMsg(debug, f'get HpenInverse use {end-start}s')
    T = Hpen @ HpenInverse @ Hpen
    # debugMsg(debug, MEqual(T, Hpen))
    lambdaPen = 16
    Q0 = HpenInverse @ Hpen
    P0 = np.identity(Q0.shape[0]) - Q0
    Henc_block = getHencBlock(blockNum, lambdaPen=16)
    # Htar_block = getHtarBlock(blockNum)
    Henc = blocks2Mat(n, Henc_block)
    end = time.time()
    debugMsg(debug, f'get Henc use {end-start}s')
    # Htar = blocks2Mat(n//2, Htar_block)
    end =time.time()
    debugMsg(debug, f"get Htar use {end-start}s")
    Heff_2nd_order = P0@Henc@P0 - (P0@Henc@Q0@HpenInverse@Q0@Henc@P0 / lambdaPen)
    # debugMsg(debug, checkSame(P @ Heff_2nd_order @ P, U@Htar@U.conj().T))
    def A2LA(A):
        return P @ A @ Q @ HpenInverse @ Q @ A @ P
    # A = pauliExpr2Mat(n, 'X1*Z4+X2*Z4')
    A = pauliExpr2Mat(n, Astring)
    la = A2LA(A)
    debugMsg(debug, LA.norm(la))
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
                #     debugMsg(debug, mj, mj_, eff, prod)
                #     debugMsg(debug, m1, m0, j1, j0, m1_, m0_, j1_, j0_)
                res += eff * prod
        return res
    plist = ('I', 'X', 'Y', 'Z')
    resultOp = []
    # exit(0)
    for i in plist:
        for j in plist:
            for k in plist:
                for l in plist:
                    s = i+j+k+l
                    result = getEff(n, s)
                    if abs(result) > 1e-4:
                        resultOp.append((s,result[0][0].real))
                        debugMsg(debug, s, result)
    return resultOp
    # n = 6
    # k = 3
    # Xeff, Zeff = (0, -1, -2, -1, 0, 1), (2, 0, -2, 0, 2, 1)
    # H = getHamil(n,Xeff,Zeff)
    # list, res = searchLogical(n, k, H)
    # debugMsg(debug, list)
    # debugMsg(debug, res)

def searchOp(getHtarBlock, targetOp, blocksize, blockNum):
    pauliPairs = [f'X{i}*Z{j}' for i in range(3) for j in range(4,8)] + [f'Z{i}*X{j}' for i in range(3) for j in range(4,8)]
    candidates = ['Z1*X6+Z3*X6+Z1*X4']
    goodPauli = None
    for c in candidates:
        result = getLogicOp(getHtarBlock, blocksize, blockNum, c)
        exits = False
        valid = True
        for item in result:
            string = item[0]
            if string == targetOp:
                exits = True
            if string.count('I')<2:
                valid=False
                break
        if valid and exits:
            goodPauli = (c, result)
            break
    return goodPauli

def palinPauliStr2IndexPauliStr(pstr):
    result = []
    for i in range(len(pstr)):
        if pstr[i]=='I':
            continue
        result.append(f'{pstr[i]}{i}')
    return '*'.join(result)

def checkOps(Ops, Astring, lamb, blockNum):
    logDict = {}
    for i in range(blockNum):
        logDict[f'X{2*i}'] = [(-1, f'X{4*i}*X{4*i+2}')]
        logDict[f'X{2*i+1}'] = [(1, f'X{4*i}*X{4*i+1}')]
        logDict[f'Z{2*i}'] = [(1, f'Z{4*i}*Z{4*i+1}')]
        logDict[f'Z{2*i+1}'] = [(1, f'Z{4*i}*Z{4*i+2}')]
        logDict[f'Z{2*i}*Z{2*i+1}'] = [(1, f'Z{4*i+1}*Z{4*i+2}')]
    HtarBlock = []
    HencBlock = []
    for op in Ops:
        if op[0].count('I')<2:
            print("there are three-local terms")
            print(Ops)
            exit(0)
        pstr = palinPauliStr2IndexPauliStr(op[0])
        if op[0].count('I')==2:
            HtarBlock.append((-op[1], pstr))
        if op[0].count('I') == 3:
            HencBlock.append((op[1], logDict[pstr][0][1]))
    # print(HtarBlock)
    # print(HencBlock)
    Htar = blocks2Mat(4, HtarBlock)
    n = 8
    Henc = lamb**0.5*pauliExpr2Mat(n, Astring) + blocks2Mat(8, HencBlock)
    Hpen = getHpen(n, 2)
    U = getU(n//2, blockNum)
    P = U @ U.conj().T
    HpenInverse = np.linalg.pinv(Hpen)
    Q0 = HpenInverse @ Hpen
    P0 = np.identity(Q0.shape[0]) - Q0
    Heff_2nd_order = P0@Henc@P0 - (P0@Henc@Q0@HpenInverse@Q0@Henc@P0 / lamb)
    flag1 = checkSame(P @ Heff_2nd_order @ P, U@Htar@U.conj().T)
    flag2 = checkSame(P @ Heff_2nd_order @ (P0-P), np.zeros(P.shape))
    return flag1, flag2

def logHadmard(logStr):
    result = ''
    Hmap = {'X':'Z', 'Z':'X', 'I':'I'}
    for i in range(len(logStr)):
        if i%2==1:
            result+=Hmap[logStr[i]]
        else:
            result+=logStr[i]
    return result

def checkAstring(Astring, lamb, blockNum):
    result = getLogicOp(getIsingHtarBlock, blocksize, blockNum, Astring)
    effs = [(logHadmard(a[0]), a[1]) for a in result]
    result = checkOps(effs, Astring, lamb, blockNum)
    return result

if __name__ =='__main__':
    Astring = 'Z1*X6+Z3*X6'
    lamb = 512
    blockNum = 2
    blocksize = 4
    result = checkAstring(Astring, lamb, blockNum)
    print(result)
    # p = searchOp(getIsingHtarBlock, 'IXZI', 4, 2)
    # print(p)