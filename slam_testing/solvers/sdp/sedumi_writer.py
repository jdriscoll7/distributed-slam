# Page 7 of https://www.ece.uvic.ca/~wslu/Talk/SeDuMi-Remarks.pdf discusses how to convert SDP problem into
# LP form of sedumi.

import numpy as np
from scipy.io import savemat
from scipy.sparse import csc_matrix, csr_matrix

from utility.parsing import parse_g2o
from solvers.sdp.matrix_creation import w_from_vertices_and_edges


def w_to_sedumi(w, output_file):
    """
    Converts W matrix into sedumi format for NEOS.

    :param w:               W matrix from Carlone paper
    :param output_file:     File name for sedumi .mat file
    :return:                nothing
    """

    # W matrix gets put into c in sedumi format.
    c = csr_matrix(w.flatten('F'))

    # Store dimension of optimization variable.
    dimension = (w.shape[0] // 2) + 1

    # Summing lambdas - vector of ones.
    b = np.ones((dimension, 1))

    # Build A matrix.
    a = csc_matrix((w.shape[0]*w.shape[0], dimension))

    # Set the 1's in the A matrix.
    a[w.shape[0] * w.shape[0] - 1, dimension - 1] = 1

    for i in range(1, dimension):
        a[w.shape[0]*w.shape[0] - (w.shape[0] + 1)*i - 1, dimension - i - 1] = 1

    # Construct K matrix for PSD constraint.
    k_s = w.shape[0]

    # Eliminate zeros to make sure each matrix isn't taking up more space than it needs.
    c.eliminate_zeros()
    a.eliminate_zeros()

    # Put everything into dictionary to save into .mat file.
    data = {"A": a, "b": b, "c": c, "K": {"s": float(k_s)}}

    # Save to mat file.
    savemat(output_file, data, do_compression=True)


if __name__ == "__main__":

    # Get w matrix.
    # vertices, edges = parse_g2o("/home/joe/repositories/distributed-slam/datasets/input_MITb_g2o.g2o")
    vertices, edges = parse_g2o("/home/joe/repositories/distributed-slam/datasets/input_INTEL_g2o.g2o")
    w = w_from_vertices_and_edges(vertices, edges)

    w_to_sedumi(w, "test.mat")
