import numpy as np
import cvxpy as cp
import scipy as sc
import copy
from multiprocessing import Pool


from solvers.experimental.fixed_sdp import vector_to_complex, rotation_vector, rotation_matrix, offset_matrix
from solvers.experimental.local_updates import LocalOptimizer
from solvers.sdp.matrix_creation import w_from_graph
from utility.visualization import plot_complex_list, plot_vertices, draw_plots


class LocalADMM(LocalOptimizer):

    def __init__(self, pgo_file=None, vertices=None, edges=None):

        # Initialize basic problem details.
        LocalOptimizer.__init__(self, pgo_file, vertices, edges)

        # Make variables (global, local, dual) initialized to original graph.
        self.global_estimate = copy.copy(self.graph)
        self.local_estimates = [self.graph.neighborhood(i, reduce=True) for i in range(len(self.graph.vertices))]

        # Only store outer product matrices for dual variables.
        self.dual_estimates = [np.zeros((2*len(self.graph.neighborhood(i).vertices) - 1, 2*len(self.graph.neighborhood(i).vertices) - 1))
                               for i in range(len(self.graph.vertices))]

        # SDP data matrices don't change between iterations. Store these for efficiency.
        self.w = [w_from_graph(self.graph.neighborhood(i, reduce=True)[0]) for i in range(len(self.graph.vertices))]

    def update_local_estimates(self, rho=1):

        pool = Pool(processes=8)
        results = pool.starmap(self.local_solve, [(i, rho) for i in range(len(self.graph.vertices))])

        for i, state in enumerate(results):

            # plot_vertices(vertices=self.local_estimates[i][0].vertices,
            #               edges=self.local_estimates[i][0].edges,
            #               labels=True)
            self.local_estimates[i][0].update_states([i for i in range(state.shape[0] // 2)], state)
            # plot_vertices(vertices=self.local_estimates[i][0].vertices,
            #               edges=self.local_estimates[i][0].edges,
            #               labels=True)

        # state = []
        # for i in range(1, len(self.graph.vertices)):
        #     print(i)
        #     state.append(self.local_solve(i))
        #
        # for i in range(1, len(self.graph.vertices)):
        #     self.local_estimates[i][0].update_states([i for i in range(state[i].shape[0] // 2)], state[i])
        #
        return

    def local_solve(self, i, rho=1):

        # Store local graph.
        graph = self.local_estimates[i][0]

        # Find SDP data matrix for this vertex.
        w = self.w[i]

        # Number of vertices in problem instance.
        n = (w.shape[0] + 1) // 2

        # Matrix optimization variable.
        X = cp.Variable(hermitian=True, shape=(2 * n - 1, 2 * n - 1))

        # Convert global and dual states into rank one matrices.
        U = cp.Constant(self.dual_estimates[i])
        z = self.global_estimate.neighborhood(i, reduce=True)[0].get_complex_state(centered=True)
        # U = cp.Constant(u @ np.conjugate(u.T))
        Z = cp.Constant(z @ np.conjugate(z.T))

        # Add positive semi-definiteness constraint.
        constraints = [X >> 0] + [X[i, i] == 1 for i in range(n - 1, 2 * n - 1)]

        # Define function f to be minimized.
        f = cp.abs(cp.trace(w @ X)) + (rho / 2 * cp.norm(X - Z + U, "fro") ** 2)
        # f = cp.abs(cp.trace(w @ X)) + (rho / 2 * cp.norm(X - Z, "fro") ** 2)
        # Form and solve problem.
        problem = cp.Problem(cp.Minimize(f), constraints)
        problem.solve(verbose=False, max_iters=100000)

        # If problem is infeasible, make no change.
        if X.value is None:
            X = graph.get_complex_state(centered=True)
        else:
            # Find rank-one approximation to program solution.
            X = rank_one_approximation(X.value)

        # For the lowest id vertex, only the rotation is updated. Place the unchanged position value
        # into the solution X to make writing changes simpler.
        X = np.vstack((np.zeros((1, 1)), X))
        # X = -X
        # X[:X.shape[0] // 2] = X[:X.shape[0] // 2] - offset

        # Find edge corresponding to anchored vertex.
        edge = [e for e in graph.edges if e.out_vertex == 0 or e.in_vertex == 0][0]

        if edge.out_vertex == 0:
            angle = np.angle(X[edge.in_vertex] - X[0]) - np.angle(vector_to_complex(edge.relative_pose))
        else:
            angle = np.angle(X[0] - X[edge.out_vertex]) - np.angle(vector_to_complex(edge.relative_pose))

        if angle > np.pi / 2:
            X = -X

        # Apply offset from anchor vertex.
        offset = vector_to_complex(graph.get_vertex(0).position)
        X[:X.shape[0] // 2] = X[:X.shape[0] // 2] + offset

        return X

    def update_global_estimate(self):

        # Number of vertices in global problem.
        n = len(self.graph.vertices)

        # Keep running sum and counts for each global index.
        sum_table = np.zeros((2*n, 2*n), dtype=np.complex)
        count_table = np.zeros((2*n, 2*n), dtype=np.complex)

        # Global estimate is just an average at each component covered by local estimates.
        for estimate in self.local_estimates:

            # Indices into submatrix determined by subgraph.
            submatrix_indices = estimate[1] + [x + n for x in estimate[1]]

            # Increment count table.
            count_table[np.ix_(submatrix_indices, submatrix_indices)] += 1

            # Form rank-one matrix to add as submatrix to global state matrix.
            state = estimate[0].get_complex_state()
            submatrix = state @ np.conj(state.T)

            # Add submatrix to global matrix.
            sum_table[np.ix_(submatrix_indices, submatrix_indices)] += submatrix

        # Perform rank-one approximation of global matrix. The resulting principle eigenvector is the global state.
        next_state = rank_one_approximation(np.divide(sum_table, count_table, where=(count_table != 0)))

        # Normalize sum with counts to compute average, and assign result to global state.
        self.global_estimate.update_states([i for i in range(next_state.shape[0])], next_state)

    def update_dual_estimates(self):

        for i, dual_estimate in enumerate(self.dual_estimates):

            # Current state as vector.
            previous_state = dual_estimate
            global_state = self.global_estimate.neighborhood(i).get_complex_state(centered=True)
            local_state = self.local_estimates[i][0].get_complex_state(centered=True)

            # Compute and set next state.
            next_state = previous_state + local_state@np.conj(local_state.T) - global_state@np.conj(global_state.T)
            self.dual_estimates[i] = next_state


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

    return sum.item()


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


def vector_angle(x, y):

    # Compute normalized inner product between two vectors.
    inner = np.dot(x / np.linalg.norm(x), y / np.linalg.norm(y))

    return np.arccos(inner)


if __name__ == "__main__":

    admm_optimizer = LocalADMM(pgo_file="/home/joe/repositories/distributed-slam/datasets/input_MITb_g2o.g2o")

    rho = 1

    for i in range(100):
        print(cost_function(admm_optimizer.global_estimate))
        admm_optimizer.update_dual_estimates()
        admm_optimizer.update_local_estimates(rho=rho)
        admm_optimizer.update_global_estimate()
        print(i)
        plot_vertices(admm_optimizer.global_estimate.vertices, labels=True)
