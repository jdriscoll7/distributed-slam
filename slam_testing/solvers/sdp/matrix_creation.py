import numpy as np
from utility.parsing import parse_g2o


def rotation_matrix_2d(theta):
    """
    Creates the basic, 2x2 rotation matrix in SO(2) determined by angle theta.

    :param theta:   counter-clockwise rotation angle
    :return:        2x2 rotation matrix represented by angle
    """

    return np.matrix([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def create_d_matrix(edges, n_vertices):
    """
    Creates the D matrix used in Carlone paper (defined in (10) and just after (11)).

    :param edges:  list of edges, each of which contains in/out-vertex information and relative pose measurements
    :return:
    """

    # D matrix is composed of smaller, 2x2 sub-matrices. Form large matrix of zeros,
    # then create and place these submatrices in proper location.
    d = np.asmatrix(np.zeros((2*len(edges), 2*n_vertices)))

    for i in range(0, 2 * len(edges), 2):

        # Get current edge.
        edge = edges[i // 2]

        # Make sure shape of relative pose is as-expected.
        relative_pose = edge.relative_pose.flatten()

        # Compute current 2x2 submatrix (D_{ij} in Carlone paper).
        submatrix = -np.matrix([[relative_pose[0], -relative_pose[1]],
                                [relative_pose[1], relative_pose[0]]])

        # Set the block matrix at block row i, block column vertex_out to submatrix.
        d[i:(i + 2), 2*edge.out_vertex:(2*edge.out_vertex + 2)] = submatrix

    return d


def create_u_matrix(edges, n_vertices):
    """
    Creates the U matrix used in Carlone paper (defined just after (12)).

    :param edges:  list of edges, each of which contains in/out-vertex information and relative pose measurements
    :return:
    """

    # Similar to construction of d matrix - just more submatrices to emplace.
    u = np.asmatrix(np.zeros((2*len(edges), 2*n_vertices)))

    for i in range(0, 2 * len(edges), 2):

        # Get current edge.
        edge = edges[i // 2]

        # Compute first submatrix (just a negative rotation matrix) - second is 2x2 identity.
        r_ij = rotation_matrix_2d(edge.rotation)

        # Set the block matrix at block row i, block column vertex_out to submatrix.
        u[i:(i + 2), 2*edge.out_vertex:(2*edge.out_vertex + 2)] = -r_ij
        u[i:(i + 2), 2*edge.in_vertex:(2*edge.in_vertex + 2)] = np.eye(2)

    return u


def edges_to_anchored_incidence(edges, n_vertices):
    """
    Converts a list of edges into the anchored incidence matrix used in Carlone W matrix.

    :param n_vertices:  number of vertices (determines number of columns)
    :param edges:       list of edges, each of which contains in/out-vertex information
    :return:            anchored incidence matrix for these edges
    """

    # Pre-allocate incidence matrix with all zeros.
    a = np.asmatrix(np.zeros((len(edges), n_vertices)))

    # Set entries of incidence matrix.
    for i in range(len(edges)):

        # Extract edge from current index.
        edge = edges[i]

        # Column pertaining to in-vertex set to +1, out-vertex set to -1.
        a[i, edge.in_vertex] = 1
        a[i, edge.out_vertex] = -1

    # Anchor the matrix (i.e. remove first column).
    return a[:, 1:]


def matrix_to_complex(a):
    """
    Converts a matrix in the scaled SO(2) group to a complex number.

    :param a: 2x2 matrix to convert to complex scalar
    :return: complex number representing matrix.
    """

    # Find scalar.
    alpha = np.sqrt(a[0, 0]**2 + a[1, 0]**2)

    # If alpha is zero, just return 0 (edge case).
    if alpha == 0:
        return 0

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


    :param a: anchored adjacency matrix for directed graph from data file
    :param d: matrix with k-th block row corresponding to edge (i,j) set to all zeros except
              i-th block column, which contains -1 * D_{ij}
    :param u: matrix with k-th block row corresponding to edge (i,j) set to all zeros except
              -1* R_{ij} and I_2 in i-th and j-th block columns respectively
    :return: W matrix used as constraint in SDP
    """

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


def w_from_g2o(path):
    """
    Creates the W matrix used in Carlone paper from g2o file.

    :param path: path to g2o data file
    :return:     W matrix used in SDP problem by Carlone
    """

    # Get edges and vertices from g2o file.
    vertices, edges = parse_g2o(path)

    # Create and return w matrix.
    return w_from_vertices_and_edges(vertices, edges)


def w_from_vertices_and_edges(vertices, edges):
    """
    Creates the W matrix fromused in Carlone paper from list of vertices and edges.

    :param vertices:    list of vertices from g2o file
    :param edges:       list of edges from g2o file
    :return:            W matrix used in SDP problem by Carlone, anchored vertex
    """

    # Create the three inputs needed to create W matrix.
    a = edges_to_anchored_incidence(edges, len(vertices))
    d = create_d_matrix(edges, len(vertices))
    u = create_u_matrix(edges, len(vertices))

    # Create and return w matrix.
    return create_w_matrix(a, d, u)


if __name__ == "__main__":

    # A = np.matrix([[0, -1, 0, -2],
    #                [1, 0, 2, 0],
    #                [0, -3, 0, -4],
    #                [3, 0, 4, 0]])
    #
    # print(complex_reduce_matrix(A).H)

    w = w_from_g2o("/home/joe/repositories/distributed-slam/datasets/input_MITb_cut.g2o")
    print(w)


