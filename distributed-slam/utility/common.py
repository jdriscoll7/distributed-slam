import numpy as np
import scipy as sc


def pd_approximation(X):

    # Return scaled, principle eigenvector of input matrix.
    w, v = sc.linalg.eigh(X)

    w[w < 0] = 0

    return v * w @ np.conj(v.T)


def rank_one_approximation(X):

    # Return scaled, principle eigenvector of input matrix.
    # w, v = sc.linalg.eigh(X, eigvals=(X.shape[0] - 2, X.shape[0] - 1))
    # return np.sum(np.sqrt(w) * v, axis=1).reshape((-1, 1))
    w, v = sc.linalg.eigh(X, eigvals=(X.shape[0] - 1, X.shape[0] - 1))
    return np.sqrt(w) * v


def vector_angle(x, y):

    # Compute normalized inner product between two vectors.
    inner = np.dot(x / np.linalg.norm(x), y / np.linalg.norm(y))

    return np.arccos(inner)


def offset_matrix(relative_pose):
    """
    Creates the D_ij matrix used in Carlone paper to simplify PGO function..

    :param relative_pose:   vector containing relative, planar measurement
    :return:                2x2 array representing offset matrix
    """

    return np.array([[relative_pose[0], -relative_pose[1]],
                     [relative_pose[1], relative_pose[0]]])


def rank_one_matrix_completion(X):

    non_zero_indices = np.where(X != 0)
    n = X.shape[0]
    y = X[non_zero_indices]

    A = np.random.normal(size=(n, 1)) + 1j*np.random.normal(size=(n, 1))
    #B = A.conj().T

    def linsolve_regular(A, b, lam=1e-4):
        return np.linalg.solve(A.conj().T @ A + lam * np.eye(A.shape[1]), A.conj().T @ b)

    losses = []
    for i in range(40):
        loss = np.mean(((A @ A.conj().T)[non_zero_indices] - y) ** 2)
        losses.append(loss)

        # Update A
        for i in range(n):
            A[i, :] = linsolve_regular(
                A.conj().T[:, non_zero_indices[1][non_zero_indices[0] == i]].T, y[non_zero_indices[0] == i])

    return A

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


def complex_to_vector(x):

    return np.vstack((np.real(x), np.imag(x)))


def rotation_to_complex(angle):

    return np.exp(1j*angle)


def cost_function(graph):

    sum = 0

    for e in graph.edges:

        in_vertex = graph.get_vertex(e.in_vertex)
        out_vertex = graph.get_vertex(e.out_vertex)

        # Difference of in and out vertex positions.
        difference = (in_vertex.position - out_vertex.position).reshape(-1, 1)

        # First term in sum.
        first_term = difference - offset_matrix(e.relative_pose) @ rotation_vector(out_vertex.rotation)

        # Second term in sum.
        second_term = rotation_vector(in_vertex.rotation) - rotation_matrix(e.rotation) @ rotation_vector(out_vertex.rotation)

        # print("\nFirst term in cost function: \t%f\nSecond term in cost function: \t%f" % (np.linalg.norm(first_term), np.linalg.norm(second_term)))

        sum += first_term.T @ first_term + second_term.T @ second_term

    return np.asarray(sum).item()
