import numpy as np
import scipy.linalg as la

from .geometry import product


class Basis:
    def __init__(self, elements, inner_product=product):
        self.elements = elements
        self.dim = len(elements)
        self.gram = np.zeros((self.dim, self.dim), dtype=np.complex128)
        self.inner_product = inner_product
        for i in range(self.dim):
            for j in range(self.dim):
                self.gram[i, j] = inner_product(self.elements[i], self.elements[j])

    def decompose(self, obj):
        rhs = np.array([self.inner_product(element, obj) for element in self.elements], dtype=np.complex128)
        return np.conj(la.solve(self.gram, rhs))

    def compose(self, vector):
        return np.sum([self.elements[i] * vector[i] for i in range(self.dim)])
