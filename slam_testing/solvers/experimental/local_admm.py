import numpy as np
import cvxpy as cp
import scipy as sc
import copy
from multiprocessing import Pool


from solvers.experimental.fixed_sdp import vector_to_complex, rotation_vector, rotation_matrix, offset_matrix
from solvers.experimental.local_updates import LocalOptimizer
from solvers.sdp.matrix_creation import w_from_graph
from utility.visualization import plot_complex_list, plot_vertices, draw_plots, plot_pose_graph
from utility.graph.data_structures import Graph


class LocalADMM(LocalOptimizer):

    def __init__(self, pgo_file=None, vertices=None, edges=None, graph=None):

        # Initialize basic problem details.
        LocalOptimizer.__init__(self, pgo_file, vertices, edges, graph)

        # Remove this later... initializes to global solution.
        # self.sdp_solve(load=True)
        # self.graph = self.sdp_solution

        # Make variables (slack, local, dual) initialized to original graph.
        self.slack = np.zeros(shape=(len(2*self.graph.vertices), 2*len(self.graph.vertices)), dtype=np.complex)
        self.local_graphs = [self.graph.neighborhood(i, reduce=True) for i in range(len(self.graph.vertices))]
        self.local_variables = [g[0].get_complex_state() @ np.conj(g[0].get_complex_state().T)
                                for g in self.local_graphs]

        # Initialize dual and w matrices.
        self.dual = []
        self.w = []

        # Create W matrices and initialize dual_estimates.
        for neighborhood, id_list in self.local_graphs:

            # Size of dual matrix depends on if anchor vertex is in corresponding neighborhood.
            # Anchoring of data matrix (W) also depends on presence of anchor vertex in neighborhood.
            anchored = False

            if 0 in id_list:
                dual_size = 2 * len(neighborhood.vertices) - 1
                anchored = True
            else:
                dual_size = 2 * len(neighborhood.vertices)

            # Store dual and data matrix.
            self.dual.append(np.zeros((dual_size, dual_size)))
            self.w.append(w_from_graph(neighborhood, anchored=anchored))

    def update_local_estimates(self, rho=1):

        pool = Pool(processes=min(8, len(self.vertices)))
        # results = pool.starmap(self.local_solve, [(i, rho) for i in range(len(self.graph.vertices))])
        results = pool.starmap(self.local_solve_pgd, [(i, rho) for i in range(len(self.graph.vertices))])
        pool.terminate()

        # Obtain and set results.
        for i, result in enumerate(results):
            self.local_variables[i] = result

    def local_solve(self, i, rho=1):

        # Store local graph.
        graph = self.local_graphs[i][0]

        # Find SDP data matrix for this vertex.
        w = self.w[i]

        # Matrix optimization variable.
        X = cp.Variable(hermitian=True, shape=w.shape)

        # Convert global and dual states into rank one matrices.
        U = cp.Constant(self.dual[i])
        #z = self.global_estimate.neighborhood(i, reduce=True)[0].get_complex_state(centered=True)
        # U = cp.Constant(u @ np.conjugate(u.T))

        # Slack variable for star-neighborhood of vertex i.
        Z = self.get_slack_submatrix(i)

        # Add positive semi-definiteness constraint.
        constraints = [X >> 0] + [X[i, i] == 1 for i in range(w.shape[0] - (w.shape[0] + 1) // 2, w.shape[0])]

        # Define function f to be minimized.
        f = cp.abs(cp.trace(w @ X)) + (rho / 2 * cp.norm(X - Z + U, "fro") ** 2)
        # f = cp.abs(cp.trace(w @ X)) + (rho / 2 * cp.norm(X - Z, "fro") ** 2)
        # Form and solve problem.
        problem = cp.Problem(cp.Minimize(f), constraints)
        problem.solve(verbose=False, max_iters=100000)

        # If problem is infeasible, make no change.
        if problem.status is not "optimal":
            X = graph.get_complex_state(centered=False)
        else:
            # Find rank-one approximation to program solution.
            X = rank_one_approximation(X.value)

        # Add anchored pose if anchor vertex contained in neighborhood.
        if 0 in self.local_graphs[i][1]:
            X = np.vstack((np.zeros((1, 1)), X))

        return X

    def local_solve_pgd(self, i, rho=1):

        # Find SDP data matrix for this vertex.
        w = self.w[i]

        # Matrix optimization variable.
        X = np.zeros(dtype=np.complex, shape=w.shape)

        # Convert global and dual states into rank one matrices.
        U = self.dual[i]

        # Slack variable for star-neighborhood of vertex i.
        Z = self.get_slack_submatrix(i)

        # Projected gradient descent.
        alpha = 0.5
        tol = 1e-3
        for _ in range(1, 1000):

            previous_X = X

            X = pgo_projection(X - alpha * (np.conj(w.T) + rho * np.conj((X - Z + U).T)))

            # Early termination condition.
            if np.linalg.norm(X - previous_X, ord='fro') < tol:
                break

        # Add anchored pose if anchor vertex contained in neighborhood.
        if 0 in self.local_graphs[i][1]:

            padded_X = np.zeros(shape=(X.shape[0] + 1, X.shape[1] + 1), dtype=np.complex)
            padded_X[1:, 1:] = X
            X = padded_X

        return X

    def update_slack(self):

        # Number of vertices in global problem.
        n = len(self.graph.vertices)

        # Keep running sum and counts for each global index.
        sum_table = np.zeros((2*n, 2*n), dtype=np.complex)
        count_table = np.zeros((2*n, 2*n), dtype=np.complex)

        # Global estimate is just an average at each component covered by local estimates.
        for i, estimate in enumerate(self.local_graphs):

            # Indices into submatrix determined by subgraph.
            submatrix_indices = estimate[1] + [x + n for x in estimate[1]]

            # Increment count table.
            count_table[np.ix_(submatrix_indices, submatrix_indices)] += 1

            # Form rank-one matrix to add as submatrix to global state matrix.
            submatrix = self.local_variables[i]

            # Add submatrix to global matrix.
            sum_table[np.ix_(submatrix_indices, submatrix_indices)] += submatrix

        # Perform PD approximation of global matrix. The resulting principle eigenvector is the global state.
        next_state = np.divide(sum_table, count_table, where=(count_table != 0))

        # Normalize sum with counts to compute average, and assign result to global state.
        self.slack = next_state

    def update_dual_estimates(self):

        for i, dual_estimate in enumerate(self.dual):

            # Current state as vector.
            previous_state = dual_estimate
            # local_state = self.local_graphs[i][0].get_complex_state()
            local_state = self.local_variables[i]

            # Compute global variable's local estimate and center.
            slack_estimate = self.get_slack_submatrix(i)

            # Center local estimate if corresponding neighborhood contains anchor vertex.
            if 0 in self.local_graphs[i][1]:
                local_state = local_state[1:, 1:]

            # Compute and set next state.
            next_state = previous_state + local_state - slack_estimate
            self.dual[i] = next_state

    def run_solver(self, iterations=1000, rho=1):

        for i in range(iterations):
            self.update_slack()
            self.update_dual_estimates()
            self.update_local_estimates(rho=rho)
            print("Iteration: %d" % i)

    def get_slack_submatrix(self, i):

        indices = self.local_graphs[i][1] + [j + self.slack[i].shape[0] // 2 for j in self.local_graphs[i][1]]

        if 0 in indices:
            indices = indices[1:]

        return self.slack[np.ix_(indices, indices)]

    def synchronize_signs(self, results):

        # Keep track of changed results.
        change_list = []

        # Determine signs for each neighborhood (by looking at the state of the center of the star-neighborhood).
        for i in range(1, len(results)):

            # All estimates corresponding to vertex i should match this sign.
            match_sign = np.sign(results[i][self.local_graphs[i][1].index(i)])
            change_list.append(i)

            for result_index, result in enumerate(results):

                # Neighborhood id's for current result and the local estimate being used to match signs.
                id_list = self.local_graphs[result_index][1]

                if i in id_list and result_index not in change_list and np.sign(result[id_list.index(i)]) != match_sign:
                    results[result_index] *= -1
                    change_list.append(result_index)

            # Early break if no more changes can be made.
            if len(change_list) == len(results):
                break

        return results

    def synchronize_angles(self, results):

        # Keep track of changed results.
        change_list = []

        # Determine signs for each neighborhood (by looking at the state of the center of the star-neighborhood).
        for i in range(1, len(results)):

            # All estimates corresponding to vertex i should match this sign.
            match_angle = np.angle(results[i][self.local_graphs[i][1].index(i)])
            change_list.append(i)

            for result_index, result in enumerate(results):

                # Neighborhood id's for current result and the local estimate being used to match signs.
                id_list = self.local_graphs[result_index][1]

                if i in id_list and result_index not in change_list and np.angle(result[id_list.index(i)]) != match_angle:
                    results[result_index] *= np.exp(1j*(match_angle - np.angle(result[id_list.index(i)])))
                    change_list.append(result_index)

            # Early break if no more changes can be made.
            if len(change_list) == len(results):
                break

        return results

    def current_estimate(self):

        graph = Graph(self.graph.vertices, self.graph.edges)
        solution = evaluate_local_solutions(self)
        graph.update_states([i for i in range(len(graph.vertices))], solution)

        print(cost_function(graph))
        # plot_pose_graph(graph=graph)

        return graph

    def augment(self, graph, vertex_id):
        """
        Adds in a single vertex and its edges without changing the local problems before the addition of vertex

        :param graph:    Full graph, including new vertex.
        :param vertex_id:   Vertex ID of new vertex.
        :return:            Nothing.
        """

        # Update full vertex list, edge list, and graph.
        self.vertices = graph.vertices
        self.edges = graph.edges
        self.graph = Graph(graph.vertices, graph.edges)

        # Neighborhood of new vertex.
        neighborhood, id_list = self.graph.neighborhood(vertex_id, reduce=True)

        # Append to list of W matrices - this is the main difference between this function and just initializing another
        # object with the full graph (adding in a vertex would change a subset of the other W matrices, this doesn't).
        anchored = True if 0 in id_list else False
        self.w.append(w_from_graph(graph=neighborhood, anchored=anchored))

        # Add neighborhood of new vertex to list of local graphs
        self.local_graphs.append((neighborhood, id_list))

        # Reset local variable values (i.e. don't start from previous estimates - tests convergence properties).
        self.local_variables = [g[0].get_complex_state() @ np.conj(g[0].get_complex_state().T)
                                for g in self.local_graphs]

        # Append new dual variable - need to add row and column of zeros on edges.
        dual_size = 2 * len(neighborhood.vertices) - (1 if anchored else 0)
        self.dual.append(np.zeros(shape=(dual_size, dual_size), dtype=np.complex))

        # Reset value of other duals back to zero matrices.
        for i, dual in enumerate(self.dual):
            self.dual[i] = np.zeros(shape=dual.shape, dtype=np.complex)

        # Reset "global" slack variable to zero matrix with one additional column and row than before.
        self.slack = np.zeros(shape=(self.slack.shape[0] + 1, self.slack.shape[1] + 1), dtype=np.complex)


def pgo_projection(X):

    Z = np.copy(X)

    for i in range(2):

        # Project onto PD cone.
        X = pd_approximation(Z)

        # Project onto constrained entries.
        Y = X
        for k in range(Y.shape[0] - (Y.shape[0] + 1) // 2, Y.shape[0]):
            Y[k, k] = 1

        # Update corrector.
        Z = Z - X + Y

    return X


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


def vector_angle(x, y):

    # Compute normalized inner product between two vectors.
    inner = np.dot(x / np.linalg.norm(x), y / np.linalg.norm(y))

    return np.arccos(inner)


def pd_approximation(X):

    # Return scaled, principle eigenvector of input matrix.
    w, v = sc.linalg.eigh(X)

    w[w < 0] = 0

    return v * w @ np.conj(v.T)


def evaluate_local_solutions(admm_optimizer):

    n = len(admm_optimizer.graph.vertices)

    solution = np.zeros(shape=(2 * n, 2 * n), dtype=np.complex)

    for i, x in enumerate(admm_optimizer.local_variables):
        indices = admm_optimizer.local_graphs[i][1] + [x + n for x in admm_optimizer.local_graphs[i][1]]
        solution[np.ix_(indices, indices)] = x

    opinion_list = [[] for _ in range(len(admm_optimizer.graph.vertices))]

    # Composed of entries from smaller, rank-one matrices.
    composite_solution = np.zeros(shape=(2 * n, 1), dtype=np.complex)

    rank_one_approximations = []
    for i, x in enumerate(admm_optimizer.local_variables):
        rank_one_approximations.append(rank_one_approximation(x))

    rank_one_approximations = admm_optimizer.synchronize_signs(results=rank_one_approximations)
    rank_one_approximations = admm_optimizer.synchronize_angles(results=rank_one_approximations)

    for i, x in enumerate(admm_optimizer.local_variables):
        indices = admm_optimizer.local_graphs[i][1] + [x + n for x in admm_optimizer.local_graphs[i][1]]
        composite_solution[indices] = rank_one_approximations[i]

    for i, x in enumerate(admm_optimizer.local_variables):
        for ii, k in enumerate(admm_optimizer.local_graphs[i][1]):
            opinion_list[k].append(rank_one_approximations[i][ii])

    mod_composite_solution = np.copy(composite_solution)
    for i, x in enumerate(opinion_list):
        # mod_composite_solution[i] = x[0]
        mod_composite_solution[i] = x[0]

    return mod_composite_solution
