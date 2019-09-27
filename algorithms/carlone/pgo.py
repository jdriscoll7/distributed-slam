import numpy as np
import cvxpy as cp
from algorithms.carlone.matrix_creation import *
from utility.parsing.g2o import parse_g2o


def w_with_multipliers(w, multipliers, n_vertices):
    """
    Computes W(lambda) as in Carlone paper. Basically subtracts a vector from the diagonal
    of the lowest-right block matrix of W. The positive semi-definiteness constraint is
    based off of this modified matrix.

    :param multipliers: vector of Lagrange multipliers
    :param n_vertices:  number of vertices in PGO
    :return:            modified w matrix used in SPD constraint
    """

    # Get indices for accessing diagonal of w.
    diag_indices = np.diag_indices(w.shape[0])

    # Get indices of diagonals to change.
    n = w.shape[0]
    indices = ([i for i in range(n_vertices - 1, n)], [i for i in range(n_vertices - 1, n)])

    # Subtract these from w.
    w[indices] = w[indices] - multipliers.value


def solve_dual_problem(w, n_vertices):
    """
    Solves (31) in the Carlone paper.

    :param n_vertices:  number of vertices in PGO problem
    :param w:           W matrix used in constraints
    :return:            lambda (solution to dual problem)
    """

    # Declare Lagrange multipliers as variable to solve dual.
    multipliers = cp.Variable((n_vertices,))

    # Add positive semi-definiteness constraint.
    constraints = [w_with_multipliers(w, multipliers, n_vertices) >> 0]

    problem = cp.Problem(cp.Maximize(cp.sum(multipliers)),
                         constraints)

    # Solve the SDP program.
    problem.solve()

    return multipliers.value


if __name__ == "__main__":

    # Get w matrix.
    vertices, edges = parse_g2o("/home/joe/repositories/distributed-slam/datasets/input_INTEL_g2o.g2o")
    w = w_from_vertices_and_edges(vertices, edges)

    # Solve.
    print(solve_dual_problem(w, len(vertices)))
