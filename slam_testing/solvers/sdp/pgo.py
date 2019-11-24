import time
import cvxpy as cp
import numpy as np
from scipy.linalg import null_space

from solvers.sdp import w_from_vertices_and_edges, w_to_sedumi
from utility.parsing import parse_g2o
from utility.visualization import plot_complex_list, plot_vertices, draw_plots
from utility.neos import neos_sdpt3_solve


def w_with_multipliers(w, multipliers):
    """
    Computes W(lambda) as in Carlone paper. Basically subtracts a vector from the diagonal
    of the lowest-right block matrix of W. The positive semi-definiteness constraint is
    based off of this modified matrix.

    :param w:           w variable in optimization problem
    :param multipliers: vector of Lagrange multipliers
    :return:            modified w matrix used in SPD constraint
    """

    # Form diagonal matrix from multipliers for lower-right block.
    d_22 = cp.diag(multipliers)

    # Correctly shape the zero block for lower-left block.
    shape = (d_22.shape[0], w.shape[1] - d_22.shape[1])
    d_21 = cp.Constant(np.zeros(shape))

    # Upper-half is all zeros - combine upper-left and upper-right blocks.
    shape = (w.shape[0] - d_22.shape[0], w.shape[1])
    d_1 = cp.Constant(np.zeros(shape))

    # Combine these blocks into one large matrix by horizontally stacking d_21 and d_22, then vertically stacking
    # the result with the top block row d_1.
    d_2 = cp.hstack([d_21, d_22])
    d = cp.vstack([d_1, d_2])

    return cp.Constant(w) - d


def solve_dual_program(w):
    """
    Solves (31) in the Carlone paper.

    :param w:           W matrix used in constraints
    :return:            lambda (solution to dual problem)
    """

    # Declare Lagrange multipliers as variable to solve dual.
    multipliers = cp.Variable(((w.shape[0] + 1) // 2,))

    # Add positive semi-definiteness constraint.
    constraints = [w_with_multipliers(w, multipliers) >> 0]

    # Form and solve problem.
    problem = cp.Problem(cp.Maximize(cp.sum(multipliers)), constraints)

    problem.solve(verbose=True)

    return multipliers.value


def solve_dual_neos(w):
    """
    Solves the large SDP in Carlone paper with NEOS server.

    :param w:   W matrix used in optimization problem.
    :return:    Optimizer of problem.
    """

    # Convert to sedumi and save locally.
    w_to_sedumi(w, "sedumi_problem.mat")

    # Send problem to NEOS and return solution.
    return neos_sdpt3_solve(input_file="sedumi_problem.mat", output_file="solution.mat")


def solve_suboptimal_program(basis):
    """
    When there is more than one zero-eigenvalue, the Carlone paper presents
    a possibly suboptimal estimate based on the solution to the dual in the
    form of a convex program. The solution of this program is the solution
    to the PGO problem in this case.

    :param basis:       basis for the null space of W(lambda) matrix, where lambda is dual solution
    :return:            vector that will be solution to PGO problem
    """

    # Vector z is the program variable.
    z = cp.Variable((basis.shape[1], 1))

    # Setup constraints (see algorithm 1 in Carlone paper for details).
    constraints = [cp.abs((basis@z)[i])**2 <= 1 for i in range((basis.shape[0] + 1) // 2, basis.shape[0])]

    # Setup objective.
    problem = cp.Problem(cp.Maximize(cp.sum(cp.real(basis@z) + cp.imag(basis@z))), constraints)

    # Solve problem.
    problem.solve(verbose=True)

    # Return solution.
    return z.value


def _pgo(w):
    """
    Implementation of algorithm 1 in Carlone paper - performs PGO given
    a W matrix, which is described in detail in paper. Code also exists
    in repository for creating W matrix from things like .g2o files.

    :param w:       large matrix used and described in Carlone paper
    :return:        solution and details
    """

    # Initialize return values.
    solution = []

    # Solve SDP with NEOS.
    print("Solving dual problem with NEOS.")
    start = time.time()
    dual_solution = solve_dual_neos(w)

    # Print resulting time elapsed.
    print("Dual problem completed. Time elapsed: %f seconds." % (time.time() - start))

    # Evaluate W(lambda).
    w_lambda = w_with_multipliers(w, dual_solution).value

    # Get eigenvalue decomposition of W(lambda).
    eigenvals, eigenvecs = np.linalg.eig(w_lambda)

    # Count number of zero eigenvalues.
    zero_multiplicity = np.sum(np.abs(eigenvals) < 1e-6)

    # If there is a single zero eigenvalue, then eigenvector corresponding to it corresponds
    # to solution.
    if zero_multiplicity == 1 or zero_multiplicity == 0:

        # Logging information.
        print("Single zero eigenvalue property holds.")

        # Find eigenvector corresponding to the single zero eigenvalue.
        v = np.asarray(eigenvecs[:, np.where(np.abs(eigenvals) == np.min(np.abs(eigenvals)))[0][0]])
        v = np.reshape(v, (-1, 1))

        # Normalize eigenvector.
        scale = np.abs([i for i in v if i != 0][-1])
        v = v / scale

        # Set solution, and update optimality certificate.
        solution = v

    else:

        # Logging information.
        print("Single zero eigenvalue property does not hold.")

        # Compute basis for null space of W(lambda) - adjust tolerance until basis isn't 0-dimensional.
        basis = null_space(w_lambda, 1e-6)

        # Solve solution to suboptimal program.
        z = solve_suboptimal_program(basis)

        # Compute optimal PGO solution based off of z, with normalization.
        x = basis@z
        x[len(eigenvals):, :] = x[len(eigenvals):, :] / np.abs(x[len(eigenvals):, :])

        # Set solution and optimality certificate.
        solution = x

    # Split solution into positions and rotations.
    positions = np.vstack([0, solution[:len(solution)//2]])
    rotations = np.vstack([0, solution[len(solution)//2:]])

    # Return solution along with optimality certificate.
    return positions, rotations, dual_solution


def pgo(x, y=None):
    """
    Wrapper for main pgo function. Can take file name or W matrix directly.

    :param x:   Either W matrix, file name, or vertices.
    :param y:   Either None or edges.
    :return:    Solution to PGO problem found with SDP.
    """

    # If y is not empty, then arguments are vertices and edges.
    if y is not None:
        return _pgo(w_from_vertices_and_edges(x, y))

    # If type is a string, then input is a file name.
    if isinstance(x, str):
        return _pgo(w_from_vertices_and_edges(*parse_g2o(x)))

    # If type is a numpy array, then the input is the W matrix.
    if isinstance(x, np.ndarray):
        return _pgo(x)

    # Raise error otherwise.
    else:
        raise TypeError


if __name__ == "__main__":

    # Generate random dataset and read it.
    # create_random_dataset(0.1, 0.1, 20, 'random_test.g2o')
    # vertices, edges = parse_g2o("random_test.g2o")
    # w = w_from_vertices_and_edges(vertices, edges)
    #
    # # Run algorithm 1 from Carlone paper.
    # solution = pgo(w)
    # plot_complex_list(solution[0][:len(vertices)])
    # plot_vertices(vertices)
    # draw_plots()

    # Get w matrix.
    vertices, edges = parse_g2o("/home/joe/repositories/distributed-slam/datasets/input_INTEL_g2o.g2o")
    # vertices, edges = parse_g2o("/home/joe/repositories/distributed-slam/datasets/input_INTEL_g2o.g2o")
    #w = w_from_vertices_and_edges(vertices, edges)

    # # Run algorithm 1 from Carlone paper.
    #solution = _pgo(w)
    solution = pgo("/home/joe/repositories/distributed-slam/slam_testing/examples/generated_data/abc.g2o")
    plot_complex_list(solution[0])
    plot_vertices(vertices)
    draw_plots()

