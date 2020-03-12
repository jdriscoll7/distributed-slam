import numpy as np
import cvxpy as cp
import scipy as sc

from solvers.sdp import w_from_g2o, pgo
from utility.graph import Graph
from utility.parsing import parse_g2o
from utility.visualization import plot_complex_list, plot_vertices, draw_plots
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


def form_quadratic(fixed_vertex, graph):

    # Vertex id's in edge-order.
    out_neighborhood = [e.out_vertex for e in graph.edges if e.in_vertex == fixed_vertex]
    in_neighborhood = [e.in_vertex for e in graph.edges if e.out_vertex == fixed_vertex]

    # Precompute all offset and rotation matrices.
    in_rotation_matrices = [rotation_matrix(e.rotation) for e in graph.edges if e.in_vertex == fixed_vertex]
    out_rotation_matrices = [rotation_matrix(e.rotation) for e in graph.edges if e.out_vertex == fixed_vertex]

    in_offset_matrices = [offset_matrix(e.relative_pose) for e in graph.edges if e.in_vertex == fixed_vertex]
    out_offset_matrices = [offset_matrix(e.relative_pose) for e in graph.edges if e.out_vertex == fixed_vertex]

    # Form upper and lower half (corresponding to positions and rotations) separately, then combine.
    a_upper = np.vstack(tuple([np.block([[-np.eye(2), -x],
                                         [np.zeros((2, 2)), -out_rotation_matrices[i]]])
                               for i, x in enumerate(in_offset_matrices)]))

    a_lower = np.vstack(tuple([np.eye(4) for _ in out_offset_matrices]))
    a = np.vstack((a_upper, a_lower))

    # Form constant vector in quadratic.
    b_upper = np.vstack(tuple([np.vstack((vector_to_complex(v.position), v.rotation))
                               for v in graph.vertices if v.id in out_neighborhood]))

    b_lower = -np.vstack(tuple([complex_reduce_matrix(np.vstack((x, in_rotation_matrices[i]))) * vector_to_complex(rotation_vector(in_neighborhood[i]))
                                for i, x in enumerate(in_offset_matrices)]))

    b = np.vstack((b_upper, b_lower))

    return complex_reduce_matrix(a), b


def create_sdp_data_matrix(a, b):
    """
    Creates data matrix for convex relaxation of quadratic program.

    :param a: "A" matrix in ||Ax + b||
    :param b: "b" vector in ||Ax + b||
    :return:  data matrix for sdp relaxation
    """

    cross_term = np.conjugate(a.T) @ b

    return np.block([[np.conjugate(a.T) @ a, cross_term],
                     [np.conjugate(cross_term.T), np.conjugate(b.T) @ b]])


def rank_one_approximation(X):

    # Return scaled, principle eigenvector of input matrix.
    w, v = sc.linalg.eigh(X, eigvals=(X.shape[0] - 1, X.shape[0] - 1))

    return np.sqrt(w) * v


def solve_local_sdp(vertex, graph):

    # Get neighborhood of vertex.
    neighborhood = graph.neighborhood(vertex)

    # Form quadratic data matrices.
    a, b = form_quadratic(vertex, neighborhood)

    # Form sdp relaxation data matrix.
    C = create_sdp_data_matrix(a, b)

    # Optimization variable.
    X = cp.Variable(hermitian=True, shape=(3, 3))

    # Setup SDP.
    constraints = [X >> 0, X[1, 1] == 1, X[2, 2] == 1]
    problem = cp.Problem(cp.Minimize(cp.abs(cp.trace(C @ X))), constraints)

    # Solve sdp.
    problem.solve(verbose=False)

    # Extract matrix solution from problem.
    problem_solution = list(problem.solution.primal_vars.values())

    # If the problem is infeasible, then make no changes.
    solution = None

    if len(problem_solution) > 0:

        # Compute rank one approximation.
        solution = rank_one_approximation(problem_solution[0])
        solution = -solution / solution[2]
        solution = [solution.item(0), solution.item(1)]

    else:

        solution = [graph.vertices[vertex].position, graph.vertices[vertex].rotation]

    return solution[0], solution[1]


if __name__ == "__main__":

    # Carlone SDP solution.
    vertices, edges = parse_g2o("../../../datasets/input_MITb_g2o.g2o")
    # positions, rotations, _ = pgo(vertices, edges)
    #
    # for i, v in enumerate(vertices):
    #     v.set_state(positions[i], rotations[i])

    # Generate pose graph matrix.
    graph = Graph(vertices, edges)

    rng = np.random.SFC64()

    plot_vertices(graph.vertices)

    for _ in range(1, 500):
        i = rng.random_raw() % (100)
        position, rotation = solve_local_sdp(i, graph)

        # Update graph with solution.
        graph.set_state(i, position, rotation)

        print(i)

    plot_vertices(graph.vertices)
    draw_plots()