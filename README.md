# QuantPy

A framework for quantum computations and quantum tomography simulations. Supports basic mathematical operations on quantum states, as well as partial traces and tensor products, measurements, quantum channels (aka CPTP maps), gates, quantum state tomography and calculating confidence intervals for the experiments.

## Installation

To install the package you need to clone this repository to your computer and install it using pip.
```bash
git clone https://github.com/esthete88/quantpy.git
cd quantpy
conda create -n quantpy --file conda-requirements
conda activate quantpy
pip install -e .
```

## Table of contents

- [Features](#features)
    - [Quantum objects](#quantum-objects)
    - [Quantum channels and gates](#quantum-channels-and-gates)
    - [Quantum state tomography](#quantum-state-tomography)
    - [Quantum process tomography](#quantum-process-tomography)

## Features

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

### Quantum channels and gates

This package offers an ability to create quantum gates and channels, as well as a collection of standard ones (Pauli gates, Hadamard, phase etc.). Gates can be easily converted into channels.
```python
rho = qp.Qobj([0.5, 0, 0, 0.5])
x_gate = qp.operator.X
x_channel = x_gate.as_channel()
rho_out = x_channel.transform(rho)
print(rho_out.bloch)
>>> array([0.5, 0, 0, 0.5])
```
Moreover, this package supports calculating Choi matrices and Kraus representations of quantum channels.
```python
channel = qp.Channel(lambda rho : qp.operator.Z @ rho @ qp.operator.Z, n_qubits=1)
print(channel.choi)
>>> Quantum object
>>>  array([[ 1.+0.j,  0.+0.j,  0.+0.j, -1.+0.j],
>>>        [ 0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j],
>>>        [ 0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j],
>>>        [-1.+0.j,  0.+0.j,  0.+0.j,  1.+0.j]])
kraus = channel.kraus
print(channel.kraus)
>>> [Quantum Operator
>>>  array([[-1.-0.j,  0.+0.j],
>>>         [ 0.+0.j,  1.+0.j]])]
```

### Quantum state tomography

With this framework it is easy to perform simulations of quantum measurements and then use the results of these simulations to reconstruct the state using different methods.
```python
import quantpy as qp


rho = qp.Qobj([1, 0], is_ket=True)
tmg = qp.StateTomograph(rho)
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

### Quantum process tomography

You can likewise perform quantum tomography of channels, as well as build confidence intervals by choosing a set of states, transforming them with channel and performing tomography on these output states. 
