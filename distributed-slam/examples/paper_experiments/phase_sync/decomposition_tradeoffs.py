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

    return positions, rotations



def solve_group_decomposed_primal(graph, groups, synchronization_vector=None):

    # Form data matrix.
    w = w_from_vertices_and_edges(graph.vertices, graph.edges, anchored=False)

    # Number of vertices in problem instance.
    n = (w.shape[0] + 1) // 2

    # Declare Lagrange multipliers as variable to solve dual.
    X = cp.Variable(hermitian=True, shape=(2 * n, 2 * n))

    constraints = [X[groups[k], :][:, groups[k]] >> 0 for k in range(len(groups))] + [X[i, i] == 1 for i in range(n - 1, 2 * n - 1)]

    # Form and solve problem.
    problem = cp.Problem(cp.Minimize(cp.abs(cp.trace(w @ X))), constraints)

    problem.solve(verbose=True, solver='SCS')

    positions, rotations = synchronize_block_solutions(solutions=[X.value[groups[i], :][:, groups[i]] for i in range(len(groups))],
                                                       graph=graph,
                                                       n_poses=n,
                                                       groups=groups,
                                                       synchronization_vector=synchronization_vector)

    graph.update_states(np.vstack((positions, rotations)))

    return graph, problem.solver_stats.solve_time


def real_data_run():

    #graph = Graph(*parse_g2o("datasets/input_INTEL_g2o.g2o"))
    graph = Graph(*parse_g2o("datasets/CSAIL_P.g2o"))

    # graph_list = form_graph_sequence(graph, 0, step=10, order="id")
    #
    # start_point = 106
    # n_graphs = 7

    graph_list = form_graph_sequence(graph, pose_id=60, step=20, order="id")

    start_point = 1
    n_graphs = 3

    solution_history = []
    changes_history = []
    solved_graphs = []

    # First solve all graphs.
    for i, g in enumerate(graph_list[start_point:(1 + start_point + n_graphs)]):
        # solution, unused = pgo(g)
        solution = solve_edge_grouped_program(g)
        solved_graphs.append(solution)

    last_graph = solved_graphs[0]

    for i, g in enumerate(graph_list[(1 + start_point):(1 + start_point + n_graphs)]):
        solution_history.append(solution)
        last_graph = solved_graphs[i]

    draw_plots()


def synthetic_data_run(n_poses=30):
    # Dataset paths.
    dataset_path = "../datasets/random_geometric_dataset.g2o"
    create_geometric_dataset(translation_variance=0.1,
                             rotation_variance=0.1,
                             n_poses=n_poses,
                             file_name=dataset_path,
                             box=[0, 10, 0, 10])

    graph = Graph(*parse_g2o(dataset_path))
    plot_pose_graph(graph)
    graph_list = form_graph_sequence(graph, 0, step=4, order="distance")

    start_point = 2
    n_graphs = 3

    solution_history = []
    changes_history = []

    solved_graphs = []

    for i, g in enumerate(graph_list[(1 + start_point):(1 + start_point + n_graphs)]):
        solution = solve_edge_grouped_program(g)
        solved_graphs.append(solution)

    for i, g in enumerate(graph_list[(1 + start_point):(1 + start_point + n_graphs)]):
        solution_history.append(solution)

    draw_plots()

def synthetic_single_run(n_poses=5):
    # Dataset paths.
    dataset_path = "../datasets/random_geometric_dataset.g2o"
    create_geometric_dataset(translation_variance=0.1,
                             rotation_variance=0.1,
                             n_poses=n_poses,
                             file_name=dataset_path,
                             box=[0, 10, 0, 10])

    graph = Graph(*parse_g2o(dataset_path))
    # plot_pose_graph(graph)

    # graph = Graph(*parse_g2o(dataset_path))
    # solution = solve_decomposed_primal(graph)
    # plt.figure()
    # plot_pose_graph(solution)

    decomposed_graph = Graph(*parse_g2o(dataset_path))
    # nonfull_solution = solve_decomposed_primal(decomposed_graph)
    d_solution, d_time = solve_edge_grouped_program(decomposed_graph)
    # plt.figure()
    # plot_pose_graph(nonfull_solution)

    primal_graph = Graph(*parse_g2o(dataset_path))
    # pgo_solution, unused = pgo(graph)
    c_solution, c_time = _solve_primal_program(primal_graph)
    # plt.figure()
    # plot_pose_graph(pgo_solution)
    # print("\nCentralized solution vector: \n " + str(pgo_solution.get_complex_state()))

    chordal_graph = Graph(*parse_g2o(dataset_path))
    chordal_solution, chordal_time = solve_chordal_program(chordal_graph)

    c_cost = cost_function(c_solution)
    d_cost = cost_function(d_solution)
    chordal_cost = cost_function(chordal_solution)
    # print("\n\n Centralized Cost: \t%f\n Decentralized Cost: \t %f" % (c_cost, d_cost))
    print("\nCentralized Cost: \t%f\n" % (c_cost,))
    print("Decentralized Cost: \t %f" % (d_cost, ))
    print("Chordal Decentralized Cost: \t %f" % (chordal_cost,))

    # draw_plots()
    return_dict = {"c_cost": c_cost,
                   "d_cost": d_cost,
                   "chordal_cost": chordal_cost,
                   "n": len(graph.edges),
                   "c_time":c_time,
                   "chordal_time": chordal_time,
                   "d_time": d_time}


    return return_dict


if __name__ == "__main__":

    # real_data_run()
    #synthetic_data_run(n_poses=5)
    costs = []

    # costs.append(synthetic_single_run(n_poses=300))
    dataset_path = "../datasets/random_geometric_dataset.g2o"
    create_geometric_dataset(translation_variance=0.1,
                             rotation_variance=0.1,
                             n_poses=30,
                             file_name=dataset_path,
                             box=[0, 10, 0, 10])
    graph = Graph(*parse_g2o(dataset_path))
    # edge_and_chordal_comparison(graph)

    for i in range(20, 100, 1):

        costs.append(synthetic_single_run(n_poses=i))

    plt.figure()
    x = np.asarray([c["n"] for c in costs])
    # c_costs = [np.log(c[1] / c[0]) for c in costs]
    # c_costs = np.log(np.asarray([c[0] for c in costs]))
    # d_costs = np.log(np.asarray([c[1] for c in costs]))
    # ch_costs = np.log(np.asarray([c[2] for c in costs]))
    c_costs = np.asarray([c["c_cost"] for c in costs])
    d_costs = np.asarray([c["d_cost"] for c in costs])
    ch_costs = np.asarray([c["chordal_cost"] for c in costs])
    c_times = np.asarray([c["c_time"] for c in costs])
    d_times = np.asarray([c["d_time"] for c in costs])
    ch_times = np.asarray([c["chordal_time"] for c in costs])
    sorted_idx = np.argsort(x)

    # Plot costs vs n_edges.
    plt.figure()
    plt.scatter(x[sorted_idx], c_costs[sorted_idx])
    plt.scatter(x[sorted_idx], d_costs[sorted_idx])
    plt.scatter(x[sorted_idx], ch_costs[sorted_idx])
    plt.legend(["Centralized", "Decentralized", "Chordal"])
    plt.ylabel("Solution Objective Value")
    plt.xlabel("# of Edges")

    # Plot time vs n_edges.
    plt.figure()
    plt.scatter(x[sorted_idx], c_times[sorted_idx])
    plt.scatter(x[sorted_idx], d_times[sorted_idx])
    plt.scatter(x[sorted_idx], ch_times[sorted_idx])
    plt.legend(["Centralized", "Decentralized", "Chordal"])
    plt.ylabel("Execution time (s)")
    plt.xlabel("# of Edges")

    print(costs)

    draw_plots()
