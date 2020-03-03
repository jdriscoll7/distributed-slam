import numpy as np
import cvxpy as cp

from solvers.sdp import w_from_g2o
from utility.graph import Graph
from utility.parsing import parse_g2o
from utility.visualization import plot_complex_list
from solvers.sdp.matrix_creation import complex_reduce_matrix


def offset_matrix(relative_pose):
    """
    Creates the D_ij matrix used in Carlone paper to simplify PGO function..

    :param relative_pose:   vector containing relative, planar measurement
    :return:                2x2 array representing offset matrix
    """

    return -np.array([[relative_pose[0], -relative_pose[1]],
                      [relative_pose[1], relative_pose[0]]])


def rotation_matrix(theta):
    """
    Creates the basic, 2x2 rotation matrix in SO(2) determined by angle theta.

    :param theta:   counter-clockwise rotation angle
    :return:        2x2 rotation matrix represented by angle
    """

    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def rotation_vector(theta):
    """
    Creates a 2x1 rotation vector determined by angle theta.

    :param theta:   counter-clockwise rotation angle
    :return:        2x1 rotation vector represented by angle
    """

    return np.array([[np.cos(theta)], [np.sin(theta)]])


def vector_to_complex(vector):

    return vector[0] + 1j*vector[1]


def form_quadratic(graph):

    # Precompute all offset and rotation matrices.
    rotation_matrices = [rotation_matrix(e.rotation) for e in graph.edges]
    offset_matrices = [offset_matrix(e.relative_pose) for e in graph.edges]

    a = np.vstack(tuple([np.block([[np.eye(2), x],
                                   [np.zeros((2, 2)), rotation_matrices[i]]])
                         for i, x in enumerate(offset_matrices)]))

    b = np.vstack(tuple([np.block([[vector_to_complex(e.relative_pose)],
                                   [np.exp(1j*e.rotation)]])
                         for e in graph.edges]))

    return complex_reduce_matrix(a), b


def create_sdp_data_matrix(a, b):
    """
    Creates data matrix for convex relaxation of quadratic program.

    :param a: "A" matrix in ||Ax + b||
    :param b: "b" vector in ||Ax + b||
    :return:  data matrix for sdp relaxation
    """

    cross_term = a.T @ b

    return np.block([[a.T @ a, cross_term],
                     [cross_term.T, b.T @ b]])


def solve_local_sdp(vertex, state, graph):

    # Get neighborhood of vertex.
    neighborhood = graph.neighborhood(vertex)

    # Form quadratic data matrices.
    a, b = form_quadratic(neighborhood)

    # Form sdp relaxation data matrix.
    C = create_sdp_data_matrix(a, b)

    # Optimization variable.
    X = cp.Variable(complex=True, shape=(3, 3))

    # Setup SDP.
    constraints = [X >> 0, X[1, 1] == 1]
    problem = cp.Problem(cp.Minimize(cp.abs(cp.trace(C @ X))), constraints)

    # Solve sdp.
    problem.solve(verbose=True)

    # Setup some constant matrices with ones on lower diagonal.
    a_22 = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    a_33 = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]])

    # Get eigenvector corresponding to lowest eigenvalue.
    solution = np.linalg.eig(C - a_22*constraints[1].dual_value - a_33*constraints[2].dual_value)[1][:, -1]

    # Scale rotation in solution and remove last entry.
    solution[1] = solution[1] / np.abs(solution[1])
    solution = solution[:2]

    return solution


if __name__ == "__main__":

    # Generate pose graph matrix.
    graph = Graph(*parse_g2o("../../../datasets/input_MITb_g2o.g2o"))

    rng = np.random.SFC64()

    for _ in range(1, 5000):
        i = rng.random_raw() % (len(graph.vertices) - 1)
        p, r = solve_local_sdp(i, [], graph)