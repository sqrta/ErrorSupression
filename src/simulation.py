from Hamil_search import *
from scipy.sparse.linalg import expm
import matplotlib.pyplot as plt
from math import log2
import sys

def EncodeTar(Htar, blockNum, lamb):
    logDict = {}
    for i in range(blockNum):
        logDict[f'X{2*i}'] = [(-1, f'X{4*i}*X{4*i+2}')]
        logDict[f'X{2*i+1}'] = [(1, f'X{4*i}*X{4*i+1}')]
        logDict[f'Z{2*i}'] = [(1, f'Z{4*i}*Z{4*i+1}')]
        logDict[f'Z{2*i+1}'] = [(1, f'Z{4*i}*Z{4*i+2}')]
        logDict[f'Z{2*i}*Z{2*i+1}'] = [(1, f'Z{4*i+1}*Z{4*i+2}')]
        logDict[f'Z{2*i+1}*Z{2*i+2}'] = [(1, f'Z{4*i+4}*Z{4*i+5}'), ((8*lamb/3)**0.5, f'Z{4*i+1}*X{4*i+6}+Z{4*i+3}*X{4*i+6}')]
    res = []
    for term in Htar:
        after = logDict[term[1]]
        for item in after:
            res.append((item[0]*term[0], item[1]))
    return res

def getU(n, blockNum):
    """
    n: number of physical qubits for each block (should be 4)
    """
    v1 = ket2Vec(n, ['0001', '-1110'])
    v2 = ket2Vec(n, ['-0010', '1101'])
    v3 = ket2Vec(n, ['0100', '-1011'])
    v4 = ket2Vec(n, ['-0111', '1000'])
    U0 = np.column_stack((v1,v2,v3,v4))
    U = U0
    for i in range(blockNum-1):
        U = np.kron(U, U0)
    return U

def getHencBlock(blockNum, lambdaPen):
    Htar_block = getIsingHtarBlock(blockNum)
    Henc_block = EncodeTar(Htar_block, blockNum, lambdaPen)
    return Henc_block

def blockize(block, blockNum, key=4):
    blocks = []
    for i in range(blockNum):
        for b in block:
            pstr = b[1]
            bmap = {}
            for j in range(key):
                bmap[f'{j}'] = f'{key*i+j}'
            for k in bmap.keys():
                pstr=pstr.replace(k, bmap[k])
            blocks.append((b[0], pstr))
    return blocks

def exp_minus_iHt(H, t):
    """
    e^{-iHt}
    """
    e, u = np.linalg.eigh(H)
    e = np.expand_dims(e, axis=1)
    A = u.conj().T
    A = np.exp(-1j*e*t) * A
    A = u @ A
    return A

def exp_minus_iHt_mult_B(H, t, B):
    """
    e^{-iHt} B
    """
    e, u = np.linalg.eigh(H)
    e = np.expand_dims(e, axis=1)
    A = u.conj().T @ B
    A = np.exp(-1j*e*t) * A
    A = u @ A
    return A

def term2Mat(size, term):
    """
    `size` means number of qubits.

    e.g. `term2Mat(4, (2, 'X1*X2'))` returns 2X1X2
    """
    return term[0] * pauliExpr2Mat(size, term[1])
    
def blocks2Mat(size, block):
    """
    `size` means number of qubits.

    e.g. `blocks2Mat(4, [(1, 'Z0*Z1'), (2, 'X1*X2'), (1, 'Z2*Z3')])` returns Z0Z1+2X1X2+Z2Z3
    """
    return sum([term2Mat(size, b) for b in block])

def getHpen(size, blockNum):
    """
    `size` means number of physical qubits.

    e.g. `getHpen(8,2)` returns X0X1+3Z0Z1+X2X3+3Z2Z3+X4X5+3Z4Z5+X6X7+3Z6Z7
    """
    Hpen_block = []
    for i in range(blockNum):
        Hpen_block += [(1, f'X{4*i}*X{4*i+1}'), (3, f'Z{4*i}*Z{4*i+1}'), (1, f'X{4*i+2}*X{4*i+3}'), (3, f'Z{4*i+2}*Z{4*i+3}')]
    Hpen = blocks2Mat(size, Hpen_block)
    return Hpen

def getError(lambdaPen, blockNum, iters=1):
    n = 4
    size = n*blockNum
    Hpen_block = []
    Htar_block = []
    for i in range(blockNum):
        Htar_block.append((1, f'Z{2*i}*Z{2*i+1}'))
        Htar_block.append((1, f'Z{2*i}'))
        Htar_block.append((1, f'Z{2*i+1}'))
        Htar_block.append((1, f'X{2*i}'))
        Htar_block.append((1, f'X{2*i+1}'))
        Hpen_block += [(1, f'X{4*i}*X{4*i+1}'), (3, f'Z{4*i}*Z{4*i+1}'), (1, f'X{4*i+2}*X{4*i+3}'), (3, f'Z{4*i+2}*Z{4*i+3}')]
    # B1 = []
    # B2 = []
    for i in range(blockNum-1):
        # B1.append((1, f'Z{2*i+1}*Z{2*i+2}'))
        # B2.append((1, f'Z{2*i+2}'))
        Htar_block.append((1, f'Z{2*i+1}*Z{2*i+2}'))
    Henc_block = EncodeTar(Htar_block, blockNum, lambdaPen)
    U = getU(n, blockNum)
    # Benc1 = EncodeTar(B1, blockNum, lambdaPen)
    # Benc2 = EncodeTar(B2, blockNum, lambdaPen)
    # HB1 = blocks2Mat(size, Benc1)
    # HB2 = blocks2Mat(size, Benc2)
    
    P = U @ U.conj().T
    # print('U', U.shape)
    # print(checkSame(P@HB1@P, P@HB2@P))
    # Q = np.identity(P.shape[0]) - P
    # print('P', P.shape)
    Henc = blocks2Mat(size, Henc_block)
    Hpen = blocks2Mat(size, Hpen_block)
    Htar = blocks2Mat(size//2, Htar_block)
    # HpenInverse = np.linalg.pinv(Hpen)
    # print('Henc', Henc.shape)
    # print('Hpen', Hpen.shape)
    # # print('Htar', Htar.shape)
    # print(checkSame(P@Hpen, np.zeros(P.shape)))
    # print(checkSame(P@Hpen@Q, np.zeros(P.shape))) 
    # print(checkSame(P@Henc@P - (P@Henc@Q@HpenInverse@Q@Henc@P / lambdaPen), U@Htar@U.conj().T))
    Hsim = lambdaPen * Hpen + Henc
    # np.random.seed(42)
    values = []
    for i in range(iters):
        epsilons = np.random.uniform(-1,1,size*3)
        V = [PauliTerm(size, f'X{i}', epsilons[i]) for i in range(size)] + [PauliTerm(size, f'Z{i}', epsilons[i+size]) for i in range(size)] + [PauliTerm(size, f'Y{i}', epsilons[i+2*size]) for i in range(size)]
        V = sum([p.value() for p in V])
        Hleft = exp_minus_iHt_mult_B(Hsim + V, t, U)
        Hright = U @ exp_minus_iHt(Htar, t)
        err = np.linalg.norm(Hleft - Hright, ord = 2)
        values.append(err)
    return values

if __name__ == '__main__':

    x = []
    blockNum = int(sys.argv[1])
    iters = int(sys.argv[2])
    t = float(sys.argv[3])
    lowBound = int(sys.argv[4])
    upBound = int(sys.argv[5])
    for i in range(lowBound, upBound+1):
        expon = i
        lamb = int(2**expon)
        local = getError(lamb, blockNum, iters)
        avg = sum(local) / len(local)
        print(lamb, avg)
        x.append((lamb, local))
    with open(f'output{blockNum}.txt', 'w') as f:
        f.write(str(x))