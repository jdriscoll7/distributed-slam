import numpy as np
import cvxpy as cp
import scipy as sc
from scipy.io import loadmat, savemat
import copy
from multiprocessing import Pool


from solvers.experimental.fixed_sdp import vector_to_complex, rotation_vector, rotation_matrix, offset_matrix
from solvers.sdp import w_from_vertices_and_edges, pgo
from solvers.sdp.matrix_creation import w_from_graph
from utility.graph import Graph
from utility.parsing import parse_g2o
from utility.visualization import plot_complex_list, plot_vertices
from solvers.sdp.pgo import solve_primal_program


class LocalOptimizer:

    def __init__(self, pgo_file=None, vertices=None, edges=None):

        # Load in vertices and edges from input arguments.
        if pgo_file is not None:
            vertices, edges = parse_g2o(pgo_file)
            self.graph = Graph(vertices, edges)

        elif vertices is not None and edges is not None:
            self.vertices = vertices
            self.edges = edges

        else:
            raise Exception("PGO file path or list of vertices and edges must be provided.")

        # Find and store anchored pose graph matrix.
        #self.w = w_from_vertices_and_edges(self.graph.vertices, self.graph.edges)

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

    def solve_local_tree(self, vertex_id):

        # Neighborhood of vertex.
        neighborhood = self.graph.neighborhood(vertex_id, reduce=False)

        for e in neighborhood.edges:
            self.graph.set_state(e.in_vertex,
                                 position=self.graph.get_vertex(e.out_vertex).position + e.relative_pose,
                                 rotation=self.graph.get_vertex(e.out_vertex).rotation + e.rotation)

    def solve_local_tree_sdp(self, vertex_id):

        # Neighborhood of vertex.
        neighborhood, vertex_ids = self.graph.neighborhood(vertex_id, reduce=True)

        # Solve SDP for tree, then find rank-one approximation to that solution.
        X = rank_one_approximation(solve_primal_program(w_from_graph(neighborhood)))

        # Translate graph appropriately (since lowest-id vertex is fixed to origin).
        offset = vector_to_complex(self.graph.get_vertex(vertex_ids[0]).position)

        # For the lowest id vertex, only the rotation is updated. Place the unchanged position value
        # into the solution X to make writing changes simpler.
        X = np.vstack((np.zeros((1, 1)), X))
        X = -X
        X[:X.shape[0] // 2] = X[:X.shape[0] // 2] - offset

        # Apply changes to graph.
        self.graph.update_states(vertex_ids, X)

        return X

    def sdp_solve(self, load=False):

        # Load precomputed solution to save time on multiple runs.
        if load:
            self.sdp_solution = np.load("pgo_solution.npy", allow_pickle=True)

        else:
            positions, rotations, _ = pgo(self.vertices, self.edges)
            self.sdp_solution = np.vstack([positions, rotations])

            # Save solution for future use.
            np.save("pgo_solution.npy", self.sdp_solution)


class LocalADMM(LocalOptimizer):

    def __init__(self, pgo_file=None, vertices=None, edges=None):

        # Initialize basic problem details.
        LocalOptimizer.__init__(self, pgo_file, vertices, edges)

        # Make variables (global, local, slack) initialized to original graph.
        self.global_estimate = copy.copy(self.graph)
        self.local_estimates = [self.graph.neighborhood(i, reduce=True) for i in range(len(self.graph.vertices))]
        self.slack_estimates = [self.graph.neighborhood(i, reduce=True) for i in range(len(self.graph.vertices))]

        # Initialize states of  estimates to zero.
        for neighborhood in self.slack_estimates:
            n = len(neighborhood[0].vertices)
            zero_state = np.zeros((2*n, 1), dtype=np.complex)
            neighborhood[0].update_states([i for i in range(n)], zero_state)

        # SDP data matrices don't change between iterations. Store these for efficiency.
        self.w = [w_from_graph(self.graph.neighborhood(i, reduce=True)[0]) for i in range(len(self.graph.vertices))]

    def update_local_estimates(self, rho=1):

        pool = Pool(processes=8)
        results = pool.starmap(self.local_solve, [(i, rho) for i in range(1, len(self.graph.vertices))])
        # for i in range(1, len(self.graph.vertices)):
        #     print(i)
        #     self.local_solve(i)

        return results

    def update_global_estimate(self):

        # Number of vertices in global problem.
        n = len(self.graph.vertices)

        # Keep running sum and counts for each global index.
        sum_table = np.zeros((2*n, 1), dtype=np.complex)
        count_table = np.zeros((2*n, 1), dtype=np.complex)

        # Global estimate is just an average at each component covered by local estimates.
        for estimate in self.local_estimates:
            for index, vertex_id in enumerate(estimate[1]):

                # Increment count table.
                count_table[vertex_id] += 1
                count_table[vertex_id + n] += 1

                # Get vertex currently being iterated on.
                vertex = estimate[0].get_vertex(index)

                # Add to running sum.
                sum_table[vertex_id] += vector_to_complex(vertex.position)
                sum_table[vertex_id + n] += vertex.rotation

        # Normalize sum with counts to compute average, and assign result to global state.
        self.global_estimate.update_states(vertex_ids=[i for i in range(n)], state=(sum_table / count_table))

    def update_slack_estimates(self, rho=1):

        for i, neighborhood in enumerate(self.slack_estimates):

            # Current state as vector.
            previous_state = neighborhood[0].get_complex_state()
            global_state = self.global_estimate.neighborhood(i).get_complex_state()
            local_state = self.local_estimates[i][0].get_complex_state()

            # Compute next state.
            next_state = previous_state + rho * (local_state - global_state)

            neighborhood[0].update_states(vertex_ids=[i for i in range(next_state.shape[0] // 2)], state=next_state)

    def local_solve(self, i, rho=1):

        print(i)

        # Neighborhood of vertex.
        neighborhood, vertex_ids = self.graph.neighborhood(i, reduce=True)

        # Find SDP data matrix for this vertex.
        w = self.w[i]

        # Number of vertices in problem instance.
        n = (w.shape[0] + 1) // 2

        # Matrix optimization variable.
        X = cp.Variable(hermitian=True, shape=(2 * n - 1, 2 * n - 1))

        # Convert global and slack states into rank one matrices.
        y = self.slack_estimates[i][0].get_complex_state(centered=True)
        z = self.global_estimate.neighborhood(i, reduce=True)[0].get_complex_state(centered=True)
        Y = cp.Constant(y @ np.conjugate(y.T))
        Z = cp.Constant(z @ np.conjugate(z.T))

        # Add positive semi-definiteness constraint.
        constraints = [X >> 0] + [X[i, i] == 1 for i in range(n - 1, 2 * n - 1)]

        # Define function f to be minimized.
        f = cp.abs(cp.trace(w @ X)) + cp.abs(cp.trace(cp.conj(Y.T) @ X)) + (rho / 2 * cp.norm(X - Z, "fro") ** 2)

        # Form and solve problem.
        problem = cp.Problem(cp.Minimize(f), constraints)
        problem.solve(verbose=False, max_iters=1000000)

        # Solve SDP for tree, then find rank-one approximation to that solution.
        X = rank_one_approximation(X.value)

        # Translate graph appropriately (since lowest-id vertex is fixed to origin).
        offset = vector_to_complex(self.graph.get_vertex(vertex_ids[0]).position)

        # For the lowest id vertex, only the rotation is updated. Place the unchanged position value
        # into the solution X to make writing changes simpler.
        X = np.vstack((np.zeros((1, 1)), X))
        # X = -X
        # X[:X.shape[0] // 2] = X[:X.shape[0] // 2] - offset
        X[:X.shape[0] // 2] = X[:X.shape[0] // 2] + offset

        # Apply changes to local solutions.
        self.local_estimates[i][0].update_states([i for i in range(n)], X)


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

        sum += first_term.T @ first_term + second_term.T @ second_term

    return sum


def rank_one_approximation(X):

    # Return scaled, principle eigenvector of input matrix.
    w, v = sc.linalg.eigh(X, eigvals=(X.shape[0] - 1, X.shape[0] - 1))

    return np.sqrt(w) * v


def test_cost_function(graph):

    x = graph.get_complex_state()
    x[0:len(graph.vertices)] = x[0:len(graph.vertices)] - x[0]
    x = x[1:]
    w = w_from_graph(graph, factored=True)

    sdp_cost = w @ x
    true_cost = cost_function(graph)

    return

if __name__ == "__main__":

    admm_optimizer = LocalADMM(pgo_file="/home/joe/repositories/distributed-slam/datasets/input_MITb_g2o.g2o")

    for i in range(2):
        print(cost_function(admm_optimizer.global_estimate))
        admm_optimizer.update_local_estimates(rho=0.01)
        admm_optimizer.update_global_estimate()
        admm_optimizer.update_slack_estimates(rho=0.01)
        print(cost_function(admm_optimizer.global_estimate))
        plot_vertices(admm_optimizer.global_estimate.vertices)


    optimizer = LocalOptimizer(pgo_file="/home/joe/repositories/distributed-slam/datasets/input_MITb_g2o.g2o")

    n = 10000
    rng = np.random.SFC64()

    optimizer.sdp_solve(load=True)

    for i in range(1, n):
        vertex_id = i % (len(optimizer.graph.vertices) - 1)#min((rng.random_raw() % len(optimizer.graph.vertices) + 1, len(optimizer.graph.vertices) - 1))
        pre_cost = cost_function(optimizer.graph.neighborhood(vertex_id))

        test_cost_function(optimizer.graph.neighborhood(vertex_id, reduce=True)[0])
        x = optimizer.solve_local_tree_sdp(vertex_id)
        test_cost_function(optimizer.graph.neighborhood(vertex_id, reduce=True)[0])

        # Testing cost.
        #W = w_from_graph(optimizer.graph.neighborhood(i, True)[0])
        #cost_compare = np.trace(W @ x)
        post_cost = cost_function(optimizer.graph.neighborhood(vertex_id))

        print("i = %d" % (i, ))