Suppose Pauli X Y Z and the identity matrices in numpy are defined as

```python
import numpy as np

X = np.array((((0, 1), (1, 0))))
Y = np.array(((0, -1j), (1j, 0)))
Z = np.array(((1, 0), (0, -1)))
I = np.identity(2)
```

suppose function `pauli2Mat` build to matrix that apply a pauli matrix `paulis` on the `indexes` qubit and `num_qubits` is the total number of system.

```python
def pauli2Mat(num_qubits, indexes, paulis):
    """
    pauli str to numpy matrices
    """
    op_list = [np.eye(2)] * num_qubits
    for index, pauli in zip(indexes, paulis):
        op_list[index] = pauli
    return tensor(op_list)
```