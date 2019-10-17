# Page 7 of https://www.ece.uvic.ca/~wslu/Talk/SeDuMi-Remarks.pdf discusses how to convert SDP problem into
# LP form of sedumi.

import numpy as np
from scipy.io import savemat
from utility.parsing.g2o import *
from algorithms.carlone.matrix_creation import *


def w_to_sedumi(w, output_file):
    """
    Converts W matrix into sedumi format for NEOS.

    :param w:               W matrix from Carlone paper
    :param output_file:     File name for sedumi .mat file
    :return:                nothing
    """

    # W matrix gets put into c in sedumi format.
    c = w.flatten('F')

    # Summing lambdas - vector of ones.
    b = np.ones((w.shape[0] // 2, 1))

    # Build A matrix.
    a = np.zeros((w.shape[0]*w.shape[0], w.shape[0] // 2))

    for i in range(0, w.shape[0] // 2):

        # Construct F_i matrix for given i.
        f_i = np.zeros((w.shape[0], w.shape[0]))

        # Find appropriate index to set to 1.
        index = (w.shape[0] // 2) + i
        f_i[index, index] = 1

        # Add F_i to A matrix.
        a[:, i] = -f_i.flatten('F')

    # Construct K matrix for PSD constraint.
    k_s = w.shape[0]

    # Put everything into dictionary to save into .mat file.
    data = {"A": a, "b": b, "c": c, "K": {"s": k_s}}

    # Save to mat file.
    savemat(output_file, data)


if __name__ == "__main__":

    # Get w matrix.
    # vertices, edges = parse_g2o("/home/joe/repositories/distributed-slam/datasets/input_MITb_g2o.g2o")
    vertices, edges = parse_g2o("/home/joe/repositories/distributed-slam/datasets/input_INTEL_g2o.g2o")
    w = w_from_vertices_and_edges(vertices, edges)

    w_to_sedumi(w, "test.mat")
