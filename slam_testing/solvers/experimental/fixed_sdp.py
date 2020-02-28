import numpy as np
import cvxpy as cp

from solvers.sdp import w_from_g2o
from utility.visualization import plot_complex_list


def solve_local_sdp(w, state, index):

    # Keep track of number of vertices in problem.
    n = (w.shape[0] + 1) // 2

    # Optimization variable.
    X = cp.Variable(complex=True, shape=(3, 3))

    # Need to set certain non-fixed variable locations to zero.
    state[index] = 0
    state[n + index] = 0

    # Form 3x3 data matrix for local sdp.
    C = np.block([[w[index, index], w[index, n + index], w[index, :] @ state],
                  [w[n + index, index], w[n + index, n + index], w[n + index, :] @ state],
                  [0, 0, 0]])

    # Setup SDP.
    constraints = [X >> 0, X[1, 1] == 1, X[2, 2] == 1]
    problem = cp.Problem(cp.Minimize(cp.abs(cp.trace(C @ X))), constraints)

    # Solve sdp.
    problem.solve(verbose=False)

    # Setup some constant matrices with ones on lower diagonal.
    a_22 = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    a_33 = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]])

    # Get eigenvector corresponding to lowest eigenvalue.
    solution = np.linalg.eig(C - a_22*constraints[1].dual_value - a_33*constraints[2].dual_value)[1][:, -1]

    # Scale rotation in solution and remove last entry.
    solution[1] = solution[1] / np.abs(solution[1])
    solution = solution[:2]

    return solution


if __name__ == "__main__":

    # Generate pose graph matrix.
    w = w_from_g2o("../../../datasets/input_MITb_g2o.g2o")

    # Compute number of vertices in problem.
    n = (w.shape[0] + 1) // 2

    # Generate random initial estimate.
    x = np.random.rand(2*n - 1, 1) + 1j*np.random.rand(2*n - 1, 1)
    x[n:] = x[n:] / np.abs(x[n:])

    rng = np.random.SFC64()

    for _ in range(1, 5000):
        i = rng.random_raw() % (n - 1)
        p, r = solve_local_sdp(w, x, i)
        x[i] = p
        x[n + i] = p

    print(1)
    plot_complex_list(x[:n])