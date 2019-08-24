# QuantPy

A framework for quantum computations and quantum tomography simulations. Supports basic operations on quantum states, measurements and quantum state tomography.

## Installation

To install the package you need to clone this repository to your computer and install it using pip.
```bash
git clone https://github.com/esthete88/quantpy.git
cd quantpy
pip install .
```

## Features

### Flexibility

You can define quantum object in 3 different ways:
- Using a matrix
```python
rho = qp.Qobj([
    [1, 0],
    [0, -1],
])
print(rho.matrix)
>>> array([[0.5, 0. ],
           [0. , 0.5]])
```
- Using a vector in the Pauli space (for Hermitian matrices)
```python
rho = qp.Qobj([0.5, 0.5, 0, 0])
print(rho.matrix)
>>> array([[0.5, 0.5],
           [0.5, 0.5]])
```
- Using a ket vector (for pure states)
```python
rho = qp.Qobj([1, 0], is_ket=True)
print(rho.matrix)
>>> array([[1., 0.],
           [0., 0.]])
```

These types are mutually connected:
```python
rho = qp.Qobj([0, 1], is_ket=True)
rho.matrix = [
    [1, 0],
    [0, 0],
]
print(rho.bloch)
>>> [0.5 0. 0. 0.5]
```

### Tensor products and partial traces

This framework supports complex operations
```python
rho_A = qp.Qobj([0.5, 0, 0, 0])
rho_B = qp.Qobj([0.5, 0, 0.5, 0])
rho_AB = rho_A.tensordot(rho_B)
tr_B = rho_AB.ptrace([0])

```

