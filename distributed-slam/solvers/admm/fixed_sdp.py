import numpy as np
import cvxpy as cp
import scipy as sc

from solvers.sdp import w_from_g2o, pgo
from utility.graph import Graph
from utility.parsing import parse_g2o
from utility.visualization import plot_complex_list, plot_vertices, draw_plots
from solvers.sdp.matrix_creation import complex_reduce_matrix


def form_quadratic(fixed_vertex, graph):

    # Vertex id's in edge-order.
    out_neighborhood = [e.out_vertex for e in graph.edges if e.in_vertex == fixed_vertex]
    in_neighborhood = [e.in_vertex for e in graph.edges if e.out_vertex == fixed_vertex]

    # Precompute all offset and rotation matrices.
    in_rotation_matrices = [rotation_matrix(e.rotation) for e in graph.edges if e.in_vertex == fixed_vertex]
    out_rotation_matrices = [rotation_matrix(e.rotation) for e in graph.edges if e.out_vertex == fixed_vertex]

    in_rotations = [np.exp(1j*v.rotation)
                    for v in graph.vertices if v.id in out_neighborhood]

    in_offset_matrices = [offset_matrix(e.relative_pose) for e in graph.edges if e.in_vertex == fixed_vertex]
    out_offset_matrices = [offset_matrix(e.relative_pose) for e in graph.edges if e.out_vertex == fixed_vertex]

    # Form upper and lower half (corresponding to positions and rotations) separately, then combine.
    a_upper = []

    for i, x in enumerate(out_offset_matrices):

        a_upper.append(np.block([[-np.eye(2), -x],
                                 [np.zeros((2, 2)), -out_rotation_matrices[i]]]))

    # Stack list of matrices vertically.
    a_upper = np.vstack(tuple(a_upper)) if len(a_upper) > 0 else []

    # Lower part of A is just 4x4 identities, stacked a number of times equal to number of incoming edges.
    if len(in_offset_matrices) > 0:
        a_lower = np.vstack(tuple([np.eye(4) for _ in in_offset_matrices]))
        if len(a_upper) > 0:
            a = np.vstack((a_upper, a_lower))
        else:
            a = a_lower
    else:
        a = a_upper


    # Form constant vector in quadratic.
    b_upper = []

    for v in graph.vertices:
        if v.id in in_neighborhood:
            b_upper.append(np.vstack((vector_to_complex(v.position), rotation_to_complex(v.rotation))))

    b_upper = np.vstack(tuple(b_upper)) if len(b_upper) > 0 else []

    # Form lower part of quadratic constant.
    b_lower = []

    for i, x in enumerate(in_offset_matrices):

        b_i = np.vstack((vector_to_complex(x @ complex_to_vector(in_rotations[i]) + graph.get_vertex(out_neighborhood[i]).position.reshape(2, 1)),
                         vector_to_complex(in_rotation_matrices[i] @ complex_to_vector(in_rotations[i]))))

        b_lower.append(-b_i)

    # Combine list of lower matrices into lower matrix, as well as combine upper and lower parts.
    if len(b_lower) > 0:
        b_lower = np.vstack(tuple(b_lower))
        if len(b_upper) > 0:
            b = np.vstack((b_upper, b_lower))
        else:
            b = b_lower
    else:
        b = b_upper

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


def solve_local_sdp(vertex, graph, verbose=False):

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
    problem.solve(verbose=verbose, max_iters=1000000)

    # Extract matrix solution from problem.
    problem_solution = list(problem.solution.primal_vars.values())

    # If the problem is infeasible, then make no changes.
    solution = None

    if len(problem_solution) > 0:

        # Compute rank one approximation.
        solution = rank_one_approximation(problem_solution[0])
        solution = solution[:, 2]
        solution = [solution.item(0), solution.item(1)]

    else:

        solution = [graph.vertices[vertex].position, graph.vertices[vertex].rotation]

    return solution[0], solution[1] / np.abs(solution[1])
