import numpy as np
import scipy.linalg as la

from .geometry import product


class Basis:
    """Class for representing a basis in the preferred Euclidean space

    Parameters
    ----------
    elements : array-like
        Basis elements
    inner_product : str or callable, default='trace'
        Inner product in the Euclidean space
        If 'trace', sets hermitian trace product in the matrix space as an inner product
            (A, B) = Tr(A @ B.H)
    """

    def __init__(self, elements, inner_product="trace"):
        self.elements = elements
        self.dim = len(elements)
        self.gram = np.zeros((self.dim, self.dim), dtype=np.complex128)
        if inner_product == "trace":
            self.inner_product = product
        else:
            self.inner_product = inner_product
        for i in range(self.dim):
            for j in range(self.dim):
                self.gram[i, j] = self.inner_product(self.elements[i], self.elements[j])

    def decompose(self, obj):
        """Return a decomposition of the obj"""
        rhs = np.array([self.inner_product(element, obj) for element in self.elements], dtype=np.complex128)
        return np.conj(la.solve(self.gram, rhs))

    def compose(self, vector):
        """Return an object using its decomposition"""
        return np.sum([self.elements[i] * vector[i] for i in range(self.dim)])

    def __repr__(self):
        return "Basis object\n" + repr(self.elements)
