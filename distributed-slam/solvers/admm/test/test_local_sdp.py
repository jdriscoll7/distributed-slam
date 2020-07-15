import numpy as np
import matplotlib.pyplot as plt
import pickle
from multiprocessing import Pool
from itertools import repeat


from solvers.admm.fixed_sdp import solve_local_sdp, rotation_matrix, rotation_vector, vector_to_complex, offset_matrix
from solvers.sdp import pgo
from utility.common import cost_function
from utility.graph import Graph
from utility.parsing import parse_g2o
from utility.visualization import plot_vertices, draw_plots


def max_cost(graph):

    max_cost = 0

    for v in graph.vertices:
        cost = cost_function(graph.neighborhood(v.id))
        if cost > max_cost:
            max_cost = cost

    return max_cost


if __name__ == "__main__":

    # Generate pose graph matrix.
    vertices, edges = parse_g2o("datasets/input_MITb_g2o.g2o")
    graph = Graph(vertices, edges)

    i = 0
    iteration = 0
    costs = dict([(v.id, []) for v in graph.vertices])

    pool = Pool(processes=10)

    while iteration < 10000:

        results = pool.starmap(solve_local_sdp, [(i, graph) for i in range(1, len(graph.vertices))])

        # i = (i + 1) % len(graph.vertices)
        #
        # pre_cost = cost_function(graph.neighborhood(i))
        #
        # if pre_cost < 1:
        #     costs[i] += [pre_cost[0][0]]
        #     continue
        #
        # old_position = vector_to_complex(graph.vertices[i].position)
        # old_rotation = vector_to_complex(rotation_vector(graph.vertices[i].rotation))[0]
        #
        # changes += 1
        # print("update: \t\t\t%d \ni: \t\t\t\t\t%d" % (changes, i))
        #
        # position, rotation = solve_local_sdp(i, graph, verbose=False)

        # Plot original neighborhood.
        #plot_vertices(graph.neighborhood(i).vertices, color='b', edges=graph.neighborhood(i).edges, labels=True)

        pre_cost = cost_function(graph)

        # Change all based on solutions, but some may be reverted if they happen to yield a higher cost.
        changes = {}

        # Update graph with solution.
        for k, result in enumerate(results):

            # Current vertex state.
            current_vertex = graph.get_vertex(k + 1)

            # Get neighborhood of vertex.
            neighborhood = graph.neighborhood(k + 1)

            # Current cost of neighborhood.
            current_cost = cost_function(neighborhood)

            # Update state and see if cost increased or decreased - revert change if increased.
            neighborhood.set_state(vertex_id=k+1, position=result[0], rotation=result[1])

            if current_cost < cost_function(neighborhood):
                changes[k + 1] = (current_vertex.position, current_vertex.rotation)
            else:
                changes[k + 1] = (result[0], result[1])

        # Make changes to global graph.
        for key, value in changes.items():
            graph.set_state(vertex_id=key, position=value[0], rotation=value[1])

        post_cost = cost_function(graph)

        print("Iteration: %d" % (iteration))
        print("Old cost: \t\t\t%f\nNew cost: \t\t\t%f\nPercent change: \t%f" % (pre_cost, post_cost, (pre_cost - post_cost) / pre_cost))

        #costs[i] += [cost_function(graph.neighborhood(i))[0][0]]

        # Plot updated neighborhood.
        #plot_vertices(graph.neighborhood(i).vertices, new_figure=True, color='r', edges=graph.neighborhood(i).edges, labels=True)
        print("\n")

        iteration += 1

    plt.figure()
    for key, value in costs.items():
        plt.plot(value)

    plot_vertices(graph.vertices, labels=True)
    draw_plots()