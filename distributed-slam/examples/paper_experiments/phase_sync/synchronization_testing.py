import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy as sc
from heapq import nlargest

from helper_functions import *
import cvxpy as cp
from collections import defaultdict

from solvers.sdp import w_from_vertices_and_edges
from utility.common import rank_one_approximation, rank_one_matrix_completion

from cvxopt import matrix, spmatrix, sparse, normal, solvers, blas, amd
import chompack
import random

np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.inf)


def nparray_to_spmatrix(A):
    coo = sc.sparse.csc_matrix(A).tocoo()
    SP = spmatrix(coo.data.tolist(), coo.row.tolist(), coo.col.tolist(), size=A.shape)
    return SP


def solve_chordal_program(graph, synchronization_vector=None):

    # Load data matrix to compute groups given by chordal decomposition.
    w = w_from_vertices_and_edges(graph.vertices, graph.edges, anchored=False)
    sparse_w = nparray_to_spmatrix(w)

    # Compute symbolic factorization of w matrix.
    symbolic_factorization = chompack.symbolic(sparse_w, p=chompack.maxcardsearch)
    cliques = symbolic_factorization.cliques(reordered=False)

    return solve_group_decomposed_primal(graph, groups=cliques, synchronization_vector=synchronization_vector)


def solve_edge_grouped_program(graph, synchronization_vector=None):

    n_vertices = len(graph.vertices)

    # Load data matrix to compute groups given by chordal decomposition.
    groups = [[e.out_vertex, e.in_vertex, e.out_vertex + n_vertices, e.in_vertex + n_vertices] for e in graph.edges]

    return solve_group_decomposed_primal(graph, groups=groups, synchronization_vector=synchronization_vector)


def _solve_primal_program(graph):
    """
    Solves (36) in the Carlone paper.

    :param w:           SDP data matrix
    :return:            lambda (solution to dual problem)
    """

    n_vertices = len(graph.vertices)

    # Load data matrix to compute groups given by chordal decomposition.
    groups = [[i for i in range(2*n_vertices)]]

    return solve_group_decomposed_primal(graph, groups=groups)

    # # Form data matrix.
    # w = w_from_vertices_and_edges(graph.vertices, graph.edges)
    #
    # # Number of vertices in problem instance.
    # n = (w.shape[0] + 1) // 2
    #
    # # Declare Lagrange multipliers as variable to solve dual.
    # X = cp.Variable(hermitian=True, shape=(2*n - 1, 2*n - 1))
    #
    # # Add positive semi-definiteness constraint.
    # constraints = [X >> 0] + [X[i, i] == 1 for i in range(n - 1, 2*n - 1)]
    #
    # # Form and solve problem.
    # problem = cp.Problem(cp.Minimize(cp.abs(cp.trace(w @ X))), constraints)
    # problem.solve(verbose=True, solver='MOSEK')
    #
    # solution = rank_one_approximation(X.value)
    #
    # positions = np.vstack([0, solution[:len(solution) // 2]])
    # rotations = solution[len(solution) // 2:]
    # graph.update_states(np.vstack((positions, rotations)))
    #
    # return graph, problem.solver_stats.solve_time


def experimental_synchronization(solution, graph, n_poses, groups):

    n_vertices = len(graph.vertices)

    # Determine rank one vectors.
    rank_one_solutions = [rank_one_approximation(solutions[i]) for i in range(len(solutions))]

    # If synchronization vector is input, we know angle sync solution. If not, calculated from local solutions.
    if synchronization_vector is None:
        # Fill out relative angle matrix.
        angle_matrix = np.zeros(shape=(2*n_vertices, 2*n_vertices), dtype=complex)

        for clique_idx, clique in enumerate(groups):
            for vertex_idx, vertex in enumerate(clique):
                parent_angle = np.angle(rank_one_solutions[clique_idx][vertex_idx])
                subvector = [entry for j, entry in enumerate(rank_one_solutions[clique_idx]) if j != vertex_idx]
                angle_differences = parent_angle - np.angle(subvector)

                parent_idx = vertex
                other_idxs = [v for v in clique if v != vertex]

                angle_matrix[parent_idx, other_idxs] = angle_differences.flatten()
                angle_matrix[other_idxs, parent_idx] = -np.copy(angle_matrix[parent_idx, other_idxs]).flatten()

        zero_entries = np.where(angle_matrix == 0)
        angle_matrix = np.exp(1j * angle_matrix)
        angle_matrix[zero_entries] = 0

        # Compute largest eigenvector and obtain angle solution from it.
        w, v = sc.linalg.eigh(angle_matrix, subset_by_index=[2*n_vertices-1, 2*n_vertices-1])

        angle_solution = np.zeros(shape=(2*n_vertices, 1), dtype=complex)
        for i in range(len(angle_solution)):
            angle_solution[i] = v[i] / (np.abs(v[i]) if np.abs(v[i]) != 0 else 1)
    else:
        angle_solution = synchronization_vector

    known_angles = {}
    for i, part in enumerate(groups):

        # Find complex angles to synchronize.
        solved_angle = angle_solution[part[0]]
        solution_angle = np.exp(1j*np.angle(rank_one_solutions[i][0]))

        # Correct the angle for the solution.
        rank_one_solutions[i] = rank_one_solutions[i] * solved_angle / solution_angle

    entry_estimates = defaultdict(list)

    for i, group in enumerate(groups):
        solution_vector = rank_one_solutions[i]
        for j, idx in enumerate(group):
            entry_estimates[idx].append(solution_vector[j])

    # Set rotations and position variables.
    positions = np.zeros(shape=(n_poses, 1), dtype=complex)
    rotations = np.zeros(shape=(n_poses, 1), dtype=complex)

    for i in range(n_poses):
        positions[i] = np.mean(entry_estimates[i])
        rotations[i] = np.mean(entry_estimates[n_poses + i])

    # Debugging.
    plt.figure()
    plot_array = [None for _ in entry_estimates]

    for k, v in entry_estimates.items():
        plot_array[k] = np.abs(np.array(v).flatten())

    plt.boxplot(plot_array)

    return positions, rotations


def synchronize_block_solutions(solutions, graph, n_poses, groups, synchronization_vector=None):

    n_vertices = len(graph.vertices)

    # Determine rank one vectors.
    rank_one_solutions = [rank_one_approximation(solutions[i]) for i in range(len(solutions))]

    # If synchronization vector is input, we know angle sync solution. If not, calculated from local solutions.
    if synchronization_vector is None:
        # Fill out relative angle matrix.
        angle_matrix = np.zeros(shape=(2*n_vertices, 2*n_vertices), dtype=complex)

        for clique_idx, clique in enumerate(groups):
            for vertex_idx, vertex in enumerate(clique):
                parent_angle = np.angle(rank_one_solutions[clique_idx][vertex_idx])
                subvector = [entry for j, entry in enumerate(rank_one_solutions[clique_idx]) if j != vertex_idx]
                angle_differences = parent_angle - np.angle(subvector)

                parent_idx = vertex
                other_idxs = [v for v in clique if v != vertex]

                angle_matrix[parent_idx, other_idxs] = angle_differences.flatten()
                angle_matrix[other_idxs, parent_idx] = -np.copy(angle_matrix[parent_idx, other_idxs]).flatten()

        zero_entries = np.where(angle_matrix == 0)
        angle_matrix = np.exp(1j * angle_matrix)
        angle_matrix[zero_entries] = 0

        # Compute largest eigenvector and obtain angle solution from it.
        w, v = sc.linalg.eigh(angle_matrix, subset_by_index=[2*n_vertices-1, 2*n_vertices-1])

        angle_solution = np.zeros(shape=(2*n_vertices, 1), dtype=complex)
        for i in range(len(angle_solution)):
            angle_solution[i] = v[i] / (np.abs(v[i]) if np.abs(v[i]) != 0 else 1)
    else:
        angle_solution = synchronization_vector

    known_angles = {}
    for i, part in enumerate(groups):

        # Find complex angles to synchronize.
        solved_angle = angle_solution[part[0]]
        solution_angle = np.exp(1j*np.angle(rank_one_solutions[i][0]))

        # Correct the angle for the solution.
        rank_one_solutions[i] = rank_one_solutions[i] * solved_angle / solution_angle

    entry_estimates = defaultdict(list)

    for i, group in enumerate(groups):
        solution_vector = rank_one_solutions[i]
        for j, idx in enumerate(group):
            entry_estimates[idx].append(solution_vector[j])

    # Set rotations and position variables.
    positions = np.zeros(shape=(n_poses, 1), dtype=complex)
    rotations = np.zeros(shape=(n_poses, 1), dtype=complex)

    for i in range(n_poses):
        positions[i] = np.mean(entry_estimates[i])
        rotations[i] = np.mean(entry_estimates[n_poses + i])

    # Debugging.
    plt.figure()
    plot_array = [None for _ in entry_estimates]

    for k, v in entry_estimates.items():
        plot_array[k] = np.abs(np.array(v).flatten())

    plt.boxplot(plot_array)

    return positions, rotations


def solve_group_decomposed_primal(graph, groups, synchronization_vector=None):

    # Form data matrix.
    w = w_from_vertices_and_edges(graph.vertices, graph.edges, anchored=False)

    # Number of vertices in problem instance.
    n = (w.shape[0] + 1) // 2

    # Declare Lagrange multipliers as variable to solve dual.
    X = cp.Variable(hermitian=True, shape=(2 * n, 2 * n))

    constraints = [X[groups[k], :][:, groups[k]] >> 0 for k in range(len(groups))] + [X[i, i] == 1 for i in range(n, 2 * n)]
    constraints += [X[0, 0] == 0]

    # Form and solve problem.
    problem = cp.Problem(cp.Minimize(cp.abs(cp.trace(w @ X))), constraints)

    problem.solve(verbose=True, solver='MOSEK')

    positions, rotations = synchronize_block_solutions(solutions=[X.value[groups[i], :][:, groups[i]] for i in range(len(groups))],
                                                       graph=graph,
                                                       n_poses=n,
                                                       groups=groups,
                                                       synchronization_vector=synchronization_vector)

    graph.update_states(np.vstack((positions, rotations)))

    return graph, problem.solver_stats.solve_time


def angular_solution_from_centralized(centralized_solution: Graph):

    solution_vector = centralized_solution.get_complex_state()
    return np.exp(1j*np.angle(solution_vector))


if __name__ == "__main__":

    results = []

    n_poses = 30
    dataset_path = "../datasets/random_geometric_dataset.g2o"
    create_geometric_dataset(translation_variance=0.1,
                             rotation_variance=0.1,
                             n_poses=n_poses,
                             file_name=dataset_path,
                             box=[0, 10, 0, 10])

    use_c_rotation = True

    graph = Graph(*parse_g2o(dataset_path))

    primal_graph = Graph(*parse_g2o(dataset_path))
    c_solution, c_time = _solve_primal_program(primal_graph)

    solution_angle = angular_solution_from_centralized(c_solution)

    decomposed_graph = Graph(*parse_g2o(dataset_path))
    d_solution, d_time = solve_edge_grouped_program(decomposed_graph, synchronization_vector=solution_angle if use_c_rotation else None)


    chordal_graph = Graph(*parse_g2o(dataset_path))
    chordal_solution, chordal_time = solve_chordal_program(chordal_graph, synchronization_vector=solution_angle if use_c_rotation else None)

    print(cost_function(c_solution))
    print(cost_function(d_solution))
    print(cost_function(chordal_solution))

    use_c_rotation = False

    graph = Graph(*parse_g2o(dataset_path))

    primal_graph = Graph(*parse_g2o(dataset_path))
    c_solution, c_time = _solve_primal_program(primal_graph)

    solution_angle = angular_solution_from_centralized(c_solution)

    decomposed_graph = Graph(*parse_g2o(dataset_path))
    d_solution, d_time = solve_edge_grouped_program(decomposed_graph, synchronization_vector=solution_angle if use_c_rotation else None)


    chordal_graph = Graph(*parse_g2o(dataset_path))
    chordal_solution, chordal_time = solve_chordal_program(chordal_graph, synchronization_vector=solution_angle if use_c_rotation else None)

    plt.figure()
    plot_pose_graph(c_solution, with_labels=True)
    plt.figure()
    plot_pose_graph(d_solution, with_labels=True)
    plt.figure()
    plot_pose_graph(chordal_solution, with_labels=True)

    print(cost_function(c_solution))
    print(cost_function(d_solution))
    print(cost_function(chordal_solution))

    plt.show()

    draw_plots()
