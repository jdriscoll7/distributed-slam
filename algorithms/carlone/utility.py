import numpy as np


def matrix_to_complex(a):
    """
    Converts a matrix in the scaled SO(2) group to a complex number.

    :param a: 2x2 matrix to convert to complex scalar
    :return: complex number representing matrix.
    """

    # Find scalar.
    alpha = np.sqrt(a[0, 0]**2 + a[1, 0]**2)

    # Find angle.
    theta = np.arctan2(a[1, 0] / alpha, a[0, 0] / alpha)

    # Return complex number.
    return alpha * np.exp(1j*theta)


def complex_reduce_matrix(a):
    """
    Converts each 2x2 block matrix in the input matrix into a single complex number,
    assuming it is composed of 2x2 matrices in the space of scaled SO(2) matrices.

    :param a: matrix composed of 2x2 block matrices
    :return: input matrix with each 2x2 block matrix converted to complex number
    """

    # Extract number of rows, columns of input.
    m, n = np.shape(a)

    # Initialize output matrix with zeros.
    output = np.zeros((m // 2, n // 2), dtype=np.complex)

    # Iterate through all 2x2 block matrices (need to step twice in each index).
    for i in range(0, m - 1, 2):
        for j in range(0, n - 1, 2):
            output[i // 2, j // 2] = matrix_to_complex(a[i:i+2, j:j+2])

    return np.asmatrix(output)


def create_w_matrix(a, d, u):
    """
    Creates the large W matrix used in the main algorithm in Carlone PGO paper (in eq (26)).

    - Anchored adjacency matrix A defined in (17).
    - D matrix defined in (10) and just after (11).
    - U matrix defined just after (12).



    :param a: non-anchored adjacency matrix for directed graph from data file
    :param d: matrix with k-th block row corresponding to edge (i,j) set to all zeros except
              i-th block column, which contains -1 * D_{ij}
    :param u: matrix with k-th block row corresponding to edge (i,j) set to all zeros except
              -1* R_{ij} and I_2 in i-th and j-th block columns respectively
    :return: W matrix used as constraint in SDP
    """

    # Cast inputs as matrix in case they are just arrays.
    a = np.asmatrix(a)
    d = np.asmatrix(d)
    u = np.asmatrix(u)

    # "Anchor" A (delete first column).
    a = a[:, 1:]

    # Pre-compute reduced d and u matrices.
    reduced_u = complex_reduce_matrix(u)
    reduced_d = complex_reduce_matrix(d)

    # Compute entries of W block-wise.
    w_11 = a.T * a
    w_12 = a.T * reduced_d
    w_21 = w_12.H
    w_22 = (reduced_u.H * reduced_u) + (reduced_d.H * reduced_d)

    return np.block([[w_11, w_12],
                     [w_21, w_22]])


if __name__ == "__main__":

    A = np.matrix([[0, -1, 0, -2],
                   [1, 0, 2, 0],
                   [0, -3, 0, -4],
                   [3, 0, 4, 0]])

    print(complex_reduce_matrix(A).H)
