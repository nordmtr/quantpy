import numpy as np
import numpy.linalg as la
import random

from numba import njit

from .qobj import Qobj
from .geometry import hs_dst


def compute_polytope_volume(polytope):
    """Compute the volume of the polytope approximately

    Parameters
    ----------
    polytope : polytope.Polytope
    """
    dim = polytope.A.shape[1]
    n_points = int(5000 * 1.5 ** dim)
    l_b, u_b = polytope.bounding_box
    x = (np.tile(l_b, (1, n_points))
         + np.random.rand(dim, n_points)
         * np.tile(u_b - l_b, (1, n_points)))
    aux = (np.dot(polytope.A, x)
           - np.tile(np.array([polytope.b]).T, (1, n_points)))
    aux = np.nonzero(np.all(aux < 0, 0))[0].shape[0]
    vol = np.prod(u_b - l_b) * aux / n_points
    return vol


@njit
def find_max_distance_to_polytope(A, b, target_point_bloch, start_point_bloch, n_points=500,
                                  discard_closer=False, hit_and_run=True):
    """Compute the distance between the target point and the farthest point in the polytope
    using hit and run algorithm.

    Parameters
    ----------
    polytope : polytope.Polytope
    """
    EPS = 1e-13
    dim = A.shape[1]
    max_dist = la.norm(start_point_bloch - target_point_bloch)

    if np.min(b - A @ start_point_bloch) < -EPS:
        return 0

    for _ in range(n_points):
        direction = np.random.rand(dim) * 2 - 1

        # discard directions pointing towards the target point
        while (discard_closer and
               np.dot(direction, start_point_bloch - target_point_bloch) <= 0):
            direction = np.random.rand(dim) * 2 - 1
        direction /= la.norm(direction)
        farthest_point_bloch = find_farthest_polytope_point(A, b, start_point_bloch, direction, EPS)
        if hit_and_run:
            theta = random.random()
            start_point_bloch = theta * start_point_bloch + (1 - theta) * farthest_point_bloch
            max_dist = max(max_dist, la.norm(start_point_bloch - target_point_bloch))
        else:
            max_dist = max(max_dist, la.norm(farthest_point_bloch - target_point_bloch))
    return max_dist * np.sqrt(np.sqrt(dim) / 2)


@njit
def find_farthest_polytope_point(A, b, start_point, direction, tol=1e-15, init_alpha=1):
    step = alpha = init_alpha
    diff_start = np.min(b - A @ start_point)
    while True:
        cur_point = start_point + alpha * direction
        diff = np.min(b - A @ cur_point)
        if diff_start < -tol:
            print(diff, step, alpha)
        if -tol <= diff < tol:
            break
        elif diff < -tol:
            step /= 2
            alpha -= step
        else:
            step *= 2
            alpha += step
    return cur_point
