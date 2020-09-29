from solvers.sdp import w_from_graph
from utility.data_generation import create_random_dataset
from utility.graph import Graph
from utility.parsing import parse_g2o
from utility.common import cost_function
from solvers.sdp.matrix_creation import complex_reduce_matrix

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat

from utility.visualization import plot_pose_graph


def block_reorder(w):

    # Form permuation vector.
    upper_indices, lower_indices = np.split(np.arange(w.shape[0]), [w.shape[0] // 2])
    permutation_vector = np.zeros(shape=(1, w.shape[0]), dtype=np.int)
    permutation_vector[:, 1::2] = upper_indices
    permutation_vector[:, ::2] = lower_indices

    # If even number, then problem is unanchored and the permutation vector needs to be shifted.
    if permutation_vector.shape[1] % 2 == 0:
        permutation_vector = np.roll(permutation_vector, -1)

    # Generate permutation matrix from permutation vector.
    permutation_matrix = np.eye(w.shape[0])
    permutation_matrix = np.squeeze(permutation_matrix[permutation_vector, :])

    new_w = np.squeeze(permutation_matrix @ w @ permutation_matrix.T)

    return new_w


def plot_sparsity_pattern(a):

    plt.subplots()
    plt.spy(a)
    plt.show()


if __name__ == "__main__":

    # Load graph from file.
    graph_path = "../../datasets/custom_five_node.g2o"
    graph = Graph(*parse_g2o(graph_path))

    # Create W matrix for graph.
    w = w_from_graph(graph, anchored=False)

    print(cost_function(graph))

    # Plot sparsity pattern.
    plot_sparsity_pattern(w)

    # Make block structure in line with Hessian structure.
    reordered_w = block_reorder(w)

    # Plot sparsity pattern again.
    plot_sparsity_pattern(reordered_w)

    plot_pose_graph(graph=graph)
    plt.show()


