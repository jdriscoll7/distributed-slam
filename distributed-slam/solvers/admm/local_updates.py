import numpy as np
import cvxpy as cp
import scipy as sc
from scipy.io import loadmat, savemat
import copy
from multiprocessing import Pool


from solvers.sdp import w_from_vertices_and_edges, pgo
from solvers.sdp.matrix_creation import w_from_graph
from utility.common import cost_function
from utility.graph import Graph
from utility.parsing import parse_g2o
from utility.visualization import plot_complex_list, plot_vertices, draw_plots
from solvers.sdp.pgo import solve_primal_program


class LocalOptimizer:

    def __init__(self, pgo_file=None, vertices=None, edges=None, graph=None):

        if graph is not None:
            vertices = graph.vertices
            edges = graph.edges

        # Load in vertices and edges from input arguments.
        if pgo_file is not None:
            self.vertices, self.edges = parse_g2o(pgo_file)

        elif vertices is not None and edges is not None:
            self.vertices = vertices
            self.edges = edges

        else:
            raise Exception("PGO file path or list of vertices and edges must be provided.")

        self.graph = Graph(self.vertices, self.edges)

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
        self.graph.update_states(X, vertex_ids)

        return X

    def sdp_solve(self, load=False):

        # Copy original graph.
        self.sdp_solution = copy.copy(self.graph)

        # Load precomputed solution to save time on multiple runs.
        if load:

            # Set states in solution graph.
            self.sdp_solution.update_states(np.load("pgo_solution.npy", allow_pickle=True))

        else:
            # Find solution.
            positions, rotations, _ = pgo(self.vertices, self.edges)

            # Set states in solution graph.
            self.sdp_solution.update_states(np.vstack([positions, rotations]))

            # Save solution for future use.
            np.save("pgo_solution.npy", self.sdp_solution)


if __name__ == "__main__":

    optimizer = LocalOptimizer(pgo_file="/home/joe/repositories/distributed-slam/datasets/input_MITb_g2o.g2o")

    n = 10000
    rng = np.random.SFC64()

    optimizer.sdp_solve(load=True)

    for i in range(1, n):
        vertex_id = i % (len(optimizer.graph.vertices) - 1)
        pre_cost = cost_function(optimizer.graph.neighborhood(vertex_id))

        x = optimizer.solve_local_tree_sdp(vertex_id)

        # Testing cost.
        #W = w_from_graph(optimizer.graph.neighborhood(i, True)[0])
        #cost_compare = np.trace(W @ x)
        post_cost = cost_function(optimizer.graph.neighborhood(vertex_id))

        print("i = %d" % (i, ))