# QuantPy

A framework for quantum computations and quantum tomography simulations. Supports basic mathematical operations on quantum states, as well as partial traces and tensor products, measurements, quantum state tomography and calculating confidence intervals for the experiments.

## Table of contents

- [Features](#features)
    - [Quantum state tomography](#quantum-state-tomography)
    - [Quantum objects](#quantum-objects)
- [Installation](#installation)

## Features

### Quantum state tomography

With this framework it is easy to perform simulations of quantum measurements and then use the results of these simulations to reconstruct the state using different methods.
```python
import quantpy as qp


rho = qp.Qobj([1, 0], is_ket=True)
tmg = qp.Tomograph(rho)
tmg.experiment(10000)  # specify the number of simulations
rho_lin = tmg.point_estimate('lin')  # linear inversion
rho_mle = tmg.point_estimate('mle')  # maximum likelihood estimation
```

The design of the framework also allows for adaptive experiments -- it is possible to continue simulations with different measurement matrices after obtaining an interim estimate of the quantum state.
```python
tmg.experiment(10000)
rho_est = tmg.point_estimate()
new_POVM = my_adaptive_func(rho_est)
tmg.experiment(10000, POVM=new_POVM, warm_start=True)
```

Moreover, you can calculate confidence intervals for these reconstructed states using built-in bootstrap functions.

### Quantum objects

You can define quantum object in 3 different ways:
- Using a matrix
```python
rho = qp.Qobj([
    [1, 0],
    [0, -1],
])
```
- Using a vector in the Pauli space (for Hermitian matrices)
```python
rho = qp.Qobj([0.5, 0.5, 0, 0])
```
- Using a ket vector (for pure states)
```python
rho = qp.Qobj([1, 0], is_ket=True)
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

## Installation

To install the package you need to clone this repository to your computer and install it using pip.
```bash
git clone https://github.com/esthete88/quantpy.git
cd quantpy
pip install .
```
