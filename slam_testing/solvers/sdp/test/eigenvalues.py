import numpy as np
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt

from solvers.sdp import pgo
from solvers.sdp.matrix_creation import w_from_g2o


def a_i(i, n):

    # Vector with 1 in i-th index.
    e_i = np.zeros((n, 1))
    e_i[i] = 1

    return e_i @ e_i.T


def coordinate_minimize(w, coordinates, passes=100):

    estimate = np.zeros((len(coordinates), ))

    # Keep track of null steps for termination condition.
    null_steps = 0

    for _ in range(passes):

        # Randomly select coordinate.
        c = np.random.choice(coordinates)

        # Get current estimate value.
        current_value = estimate[len(estimate) - c]

        # Maximize coordinate value while keeping minium eigenvalue at least zero.
        grid = np.arange(-10 + current_value, current_value + 10, 0.01)
        min_eigenvalues = np.zeros(len(grid))

        # Apply the effects from all other, fixed coordinate values. Then solve for current coordinate.
        current_w = w
        for cc in [_ for _ in coordinates if _ != c]:
            estimate_weight = estimate[len(estimate) - cc]
            current_w = current_w - estimate_weight * a_i(cc, w.shape[0])

        for index, t in np.ndenumerate(grid):
            min_eigenvalues[index] = eigh(current_w - (t * a_i(c, w.shape[0])), eigvals_only=True, eigvals=(0, 0))[0]

        # Get rid of invalid entries (negative smallest eigenvalue).
        invalid_indices = np.logical_or(min_eigenvalues < 0, min_eigenvalues < np.max(min_eigenvalues) / 1.5)
        min_eigenvalues[invalid_indices] = np.max(min_eigenvalues) + 1

        # Get update value.
        update_value = grid[np.argmin(min_eigenvalues)]

        # Apply update if update value is larger than current estimate.
        if update_value > current_value:
            print("Coordinate %d updated from %f to %f.\n" % (c, current_value, update_value))
            estimate[len(estimate) - c] = grid[np.argmax(min_eigenvalues)]
            null_steps = 0
        else:
            print("Coordinate %d null step.\n" % (c))
            null_steps += 1

        if null_steps == len(coordinates):
            break

    return estimate


if __name__ == "__main__":

    # Get W matrix - dual value for the solution is 41.6.
    w = w_from_g2o("data/dataset.g2o")

    pgo(w)

    # Eigenvalue decomposition.
    L, V = eigh(w)

    indices = [i for i in range(10, 19)]

    print(sum(coordinate_minimize(w, indices)))

    min_eigenvalues = []

    for x in np.arange(-500, 500, 0.1):

        min_eigenvalues.append(eigh(w - (x * a_i(12, 19)), eigvals_only=True, eigvals=(0, 0))[0])

    plt.figure()
    plt.plot(np.arange(-500, 500, 0.1), min_eigenvalues)
    plt.show()

