import matplotlib.pyplot as plt
from matplotlib.pyplot import spy
from chompack import symbolic
from cvxopt import matrix, sparse
from numpy import array

from solvers.sdp.matrix_creation import w_from_g2o, w_from_vertices_and_edges
from utility.data_generation import create_random_dataset


def plot_sparsity_pattern(x):

    if isinstance(x, str):
        w = w_from_g2o(x)
    else:
        w = w_from_vertices_and_edges(*x)

    spy(w)
    plt.show()


def chordal_sparsity(x):

    if isinstance(x, str):
        w = w_from_g2o(x)
    else:
        w = w_from_vertices_and_edges(*x)

    w = sparse(matrix(w))
    spy(array(matrix(symbolic(w).sparsity_pattern())))
    plt.show()


if __name__ == "__main__":

    random_data = create_random_dataset(2, 3, 10, 'dataset.g2o')
    real_data = "/home/joe/repositories/distributed-slam/datasets/input_INTEL_g2o.g2o"

    # W matrix from generated data.
    # plot_sparsity_pattern(random_data)
    # chordal_sparsity(random_data)

    # W matrix from real-world, large example.
    plot_sparsity_pattern(real_data)
    chordal_sparsity(real_data)
    print("")

