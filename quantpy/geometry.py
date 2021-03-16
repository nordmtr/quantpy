import numpy as np
import scipy.linalg as la


def hs_dst(A, B):
    """Return Hilbert-Schmidt distance between two matrices

    Parameters
    ----------
    A, B : numpy 2-D arrays or Qobjs
    """
    if not isinstance(A, np.ndarray):
        A = A.matrix
    if not isinstance(B, np.ndarray):
        B = B.matrix
    dist = np.sqrt(abs(np.trace((A - B) @ (A - B)))) / np.sqrt(2)
    if dist < 1e-15:
        return 0
    else:
        return dist


def trace_dst(A, B):
    """Return trace distance between two matrices or Qobjs

    Parameters
    ----------
    A, B : numpy 2-D arrays or Qobjs
    """
    if not isinstance(A, np.ndarray):
        A = A.matrix
    if not isinstance(B, np.ndarray):
        B = B.matrix
    dist = abs(np.trace(la.sqrtm((A - B) @ (A - B)))) / 2
    if dist < 1e-15:
        return 0
    else:
        return dist


def if_dst(A, B):
    """Return infidelity between two matrices or Qobjs

    Parameters
    ----------
    A, B : numpy 2-D arrays or Qobjs
    """
    if not isinstance(A, np.ndarray):
        A = A.matrix
    if not isinstance(B, np.ndarray):
        B = B.matrix
    dist = 1 - np.abs(np.trace(la.sqrtm(la.sqrtm(A) @ B @ la.sqrtm(A))) ** 2)
    if dist < 1e-15:
        return 0
    else:
        return dist


def product(A, B):
    """Return hermitian inner product in matrix space or Qobjs

    Parameters
    ----------
    A, B : numpy 2-D arrays or Qobjs
    """
    if not isinstance(A, np.ndarray):
        A = A.matrix
    if not isinstance(B, np.ndarray):
        B = B.matrix
    return np.trace(A @ np.conj(B.T), dtype=np.complex128)
