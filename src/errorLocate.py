import copy

def stateMap(state, op):
    maps = None
    sign = 1
    if op[0].upper()=='X':
        if op[1] % 2 == 0:
            maps = {4:(1, -2), 2:(-1, -4), -2:(1, 4), -4: (-1, 2)}
        else:
            maps = {4:(1, -2), 2:(1, -4), -2:(1, 4), -4: (1, 2)}
    elif op[0].upper()=='Z':
        if op[1] % 2 == 0:
            maps = {4:(1, 2), 2:(1, 4), -2:(1, -4), -4: (1, -2)}
        else:
            maps = {4:(1, 2), 2:(1, 4), -2:(-1, -4), -4: (-1, -2)}
    res = maps[state][1]
    sign = maps[state][0]
    return res, sign

class leakState:
    def __init__(self, value, sign, frac) -> None:
        self.value = value
        self.sign = sign
        self.frac = frac

    def applySingle(self, op):
        newValue = copy.deepcopy(self.value)
        newSign = self.sign
        index = op[1] // 2
        res, sign = stateMap(newValue[index], op)
        newValue[index] = res
        newSign *= sign
        return leakState(newValue, newSign, self.frac)
    
    def apply(self, ops):
        tmp = self
        for op in ops:
            tmp = tmp.applySingle(op)
        return tmp
    
    def setFrac(self):
        self.frac = sum(self.value)

    def eff(self):
        return self.sign/self.frac
    
    def __str__(self) -> str:
        return f"{1}/{self.frac//self.sign} * |{','.join([str(i) for i in self.value])}>"
    
def multiply(states, ops, setFrac = False):
    resultState = []
    for state in states:
        for op in ops:
            tmp = state.apply(op)
            if setFrac:
                tmp.setFrac()
            resultState.append(tmp)
    return resultState

def simplify(states):
    effMap = {}
    for state in states:
        k = tuple(state.value)
        if k in effMap.keys():
            effMap[k] += state.eff()
        else:
            effMap[k] = state.eff()
    result = []
    for k in effMap.keys():
        if abs(effMap[k])>1e-4:
            result.append(leakState(k, 1, 1/effMap[k]))
    return result

def Join(string, array):
    return string.join([str(i) for i in array])
    
if __name__ == '__main__':
    init = [-4, 4, -4, 4, 4, -4]
    initState = [leakState(init, 1, 1)]
    # opLeft = [[('Z', 1), ('X', 7)], [('Z', 3), ('X', 7)]]
    # opRight = [[('Z', 5), ('X', 11)], [('Z', 7), ('X', 11)]]
    opLeft = [[('Z', 1), ('X', 7)]]
    opRight = [[('Z', 7), ('X', 11)]]
    # opLeft = [[('Z', 1), ('X', 6)], [('Z', 3), ('X', 6)]]
    # opRight = [[('Z', 5), ('X', 10)], [('Z', 7), ('X', 10)]]
    res = multiply(initState, opLeft, setFrac=True)
    res = multiply(res, opRight)
    print(Join(' + ', res))
    res2 = multiply(initState, opRight, setFrac=True)
    res2 = multiply(res2, opLeft)
    print(Join(' + ', res2))
    
    res = simplify(res+res2)
    print(Join(' + ', res))