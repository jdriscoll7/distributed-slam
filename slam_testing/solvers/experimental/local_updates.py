import numpy as np
import cvxpy as cp
from scipy.io import loadmat, savemat

from solvers.sdp import w_from_vertices_and_edges, pgo
from utility.parsing import parse_g2o
from utility.visualization import plot_complex_list


class LocalOptimizer:

    def __init__(self, pgo_file=None, vertices=None, edges=None):

        # Load in vertices and edges from input arguments.
        if pgo_file is not None:
            self.vertices, self.edges = parse_g2o(pgo_file)
        elif vertices is not None and edges is not None:
            self.vertices = vertices
            self.edges = edges
        else:
            raise Exception("PGO file path or list of vertices and edges must be provided.")

        # Initialize to all zeros vector (this is the estimate of poses to be found). First pose is fixed.
        self.estimate = np.ones((2*len(self.vertices), 1), dtype=np.complex)

        # Find and store anchored pose graph matrix.
        self.w = w_from_vertices_and_edges(self.vertices, self.edges)

        # Variable to compare iterative solutions with sdp solution.
        self.sdp_solution = None

    def solve_local_subproblem(self, vertex_id):

        position_estimate, rotation_estimate = partial_solve(self.w, self.estimate, vertex_id)

        self.estimate[vertex_id] = position_estimate
        self.estimate[2*vertex_id] = rotation_estimate

    def get_full_estimate(self):

        return self.estimate

    def get_location_estimate(self):

        return self.estimate[:self.estimate.size // 2]

    def plot_poses(self):

        plot_complex_list(self.get_location_estimate())

    def sdp_solve(self, load=False):

        # Load precomputed solution to save time on multiple runs.
        if load:

            self.sdp_solution = np.load("pgo_solution.npy", allow_pickle=True)

        else:

            positions, rotations, _ = pgo(self.vertices, self.edges)
            self.sdp_solution = np.vstack([positions, rotations])

            # Save solution for future use.
            np.save("pgo_solution.npy", self.sdp_solution)


def partial_solve(w, fixed, vertex_id):

    x = cp.Variable(complex=True)
    p = cp.Variable(complex=True)

    stacked_positions = cp.vstack([fixed[1:vertex_id], cp.reshape(x, (1, 1)), fixed[vertex_id+1:fixed.size//2]])
    stacked_rotations = cp.vstack([fixed[fixed.size//2:fixed.size//2 + vertex_id], cp.reshape(p, (1, 1)), fixed[fixed.size//2 + vertex_id +1:]])

    stacked_vector = cp.vstack([stacked_positions, stacked_rotations])

    objective = cp.Minimize(cp.quad_form(stacked_vector, w))
    constraints = [cp.abs(p) <= 1]
    problem = cp.Problem(objective, constraints)

    # X = cp.Variable((fixed.size - 1, fixed.size - 1), symmetric=True)
    #
    # objective = cp.Minimize(cp.real(cp.trace(w @ X)))
    # constraints = [X[vertex_id, vertex_id] == 1, X >> 0]
    problem = cp.Problem(objective, constraints)

    result = problem.solve(verbose=True)

    return x.value, p.value / np.abs(p.value)

if __name__ == "__main__":

    optimizer = LocalOptimizer(pgo_file="/home/joe/repositories/distributed-slam/datasets/input_MITb_g2o.g2o")

    n = 10000
    rng = np.random.SFC64()

    optimizer.sdp_solve(load=True)

    for i in range(n):
        vertex_id = min((rng.random_raw() % len(optimizer.vertices) + 1, len(optimizer.vertices) - 1))
        optimizer.solve_local_subproblem(vertex_id)
        print("Distance from optimal: %f" % (np.max(np.abs(optimizer.estimate - optimizer.sdp_solution))))