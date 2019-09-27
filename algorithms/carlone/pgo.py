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

    # Form diagonal matrix from multipliers for lower-right block.
    d_22 = cp.diag(multipliers)

    # Correctly shape the zero block for lower-left block.
    shape = (d_22.shape[0], w.shape[1] - d_22.shape[1])
    d_21 = cp.Constant(np.zeros(shape))

    # Upper-half is all zeros - combine upper-left and upper-right blocks.
    shape = (w.shape[0] - d_22.shape[0], w.shape[1])
    d_1 = cp.Constant(np.zeros(shape))

    # Combine these blocks into one large matrix by horizontally stacking d_21 and d_22, then vertically stacking
    # the result with the top block row d_1.
    d_2 = cp.hstack([d_21, d_22])
    d = cp.vstack([d_1, d_2])

    return cp.Constant(w) - d


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

    # Form and solve problem.
    problem = cp.Problem(cp.Maximize(cp.sum(multipliers)), constraints)
    problem.solve()

    return multipliers.value


if __name__ == "__main__":

    # Get w matrix.
    vertices, edges = parse_g2o("/home/joe/repositories/distributed-slam/datasets/input_INTEL_g2o.g2o")
    w = w_from_vertices_and_edges(vertices, edges)

    # Solve.
    print(solve_dual_problem(w, len(vertices)))
