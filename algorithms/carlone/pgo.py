import numpy as np
import cvxpy as cp
from algorithms.carlone.matrix_creation import *
from utility.parsing.g2o import parse_g2o
from scipy.linalg import null_space
from scipy.io import loadmat, savemat
import time
from utility.visualization.plot import plot_complex_list
import utility.sdpt3glue as sdpt3glue


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

    sdpt3glue.sdpt3_solve_problem(problem, sdpt3glue.NEOS, "matfile.mat", "log.log")






    problem.solve(verbose=True)

    savemat("dual_solution.mat", {"dual_solution": multipliers.value})

    return multipliers.value


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
    problem.solve(verbose=True, solver=cp.ECOS)

    # Return solution.
    return z.value


def pgo(w):
    """
    Implementation of algorithm 1 in Carlone paper - performs PGO given
    a W matrix, which is described in detail in paper. Code also exists
    in repository for creating W matrix from things like .g2o files.

    :param w: large matrix used and described in Carlone paper
    :return:  solution and certificate of global optimality ("true" or "unknown")
    """

    # Initialize return values.
    solution = []
    is_optimal = ""

    # Solve SDP to find dual solution - time how long it takes as well.
    print("Solving dual problem.")
    start = time.time()
    dual_solution = solve_dual_program(w)
    #dual_solution = loadmat("dual_solution.mat")["dual_solution"]
    print("Dual problem completed. Time elapsed: %f seconds." % (time.time() - start))

    # Evaluate W(lambda).
    w_lambda = w_with_multipliers(w, dual_solution).value

    # Get eigenvalue decomposition of W(lambda).
    eigenvals, eigenvecs = np.linalg.eig(w_lambda)

    # Count number of zero eigenvalues.
    tolerance = 1e-6
    zero_multiplicity = len(eigenvals[np.abs(eigenvals) < tolerance])

    # If there is a single zero eigenvalue, then eigenvector corresponding to it corresponds
    # to solution.
    if zero_multiplicity == 1:

        # Logging information.
        print("Single zero eigenvalue property holds.")

        # Find eigenvector corresponding to the single zero eigenvalue.
        v = np.asarray(eigenvecs[:, np.where(np.abs(eigenvals) < 1e-8)[0][0] - 1])
        v = np.reshape(v, (-1, 1))

        # Normalize eigenvector.
        v[v.shape[0] // 2:, :] = v[v.shape[0] // 2:, :] / np.abs(v[v.shape[0] // 2:, :])

        # Set solution, and update optimality certificate.
        solution = v
        is_optimal = "true"

    else:

        # Logging information.
        print("Single zero eigenvalue property does not hold.")

        # Compute basis for null space of W(lambda).
        basis = null_space(w_lambda, 1e-4)

        # Solve solution to suboptimal program.
        z = solve_suboptimal_program(basis)

        # Compute optimal PGO solution based off of z, with normalization.
        x = basis@z
        x[len(eigenvals):, :] = x[len(eigenvals):, :] / np.abs(x[len(eigenvals):, :])

        # Set solution and optimality certificate.
        solution = x
        is_optimal = "unknown"

    # Return solution along with optimality certificate.
    return solution, is_optimal


if __name__ == "__main__":

    # Get w matrix.
    # vertices, edges = parse_g2o("/home/joe/repositories/distributed-slam/datasets/input_MITb_g2o.g2o")
    vertices, edges = parse_g2o("/home/joe/repositories/distributed-slam/datasets/input_INTEL_cut.g2o")
    w = w_from_vertices_and_edges(vertices, edges)

    # Run algorithm 1 from Carlone paper.
    solution = pgo(w)
    plot_complex_list(solution[0][0:199])

