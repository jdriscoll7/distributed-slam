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
from utility.visualization import plot_complex_list, plot_vertices, draw_plots
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

    optimizer = LocalOptimizer(pgo_file="/home/joe/repositories/distributed-slam/datasets/input_MITb_g2o.g2o")

    n = 10000
    rng = np.random.SFC64()

    optimizer.sdp_solve(load=True)

    for i in range(1, n):
        vertex_id = i % (len(optimizer.graph.vertices) - 1)
        pre_cost = cost_function(optimizer.graph.neighborhood(vertex_id))

        test_cost_function(optimizer.graph.neighborhood(vertex_id, reduce=True)[0])
        x = optimizer.solve_local_tree_sdp(vertex_id)
        test_cost_function(optimizer.graph.neighborhood(vertex_id, reduce=True)[0])

        # Testing cost.
        #W = w_from_graph(optimizer.graph.neighborhood(i, True)[0])
        #cost_compare = np.trace(W @ x)
        post_cost = cost_function(optimizer.graph.neighborhood(vertex_id))

        print("i = %d" % (i, ))