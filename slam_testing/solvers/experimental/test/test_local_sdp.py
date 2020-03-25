import numpy as np

from solvers.experimental.fixed_sdp import solve_local_sdp, rotation_matrix, rotation_vector, vector_to_complex, offset_matrix
from utility.graph import Graph
from utility.parsing import parse_g2o
from utility.visualization import plot_vertices, draw_plots


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


def max_cost(graph):

    max_cost = 0

    for v in graph.vertices:
        cost = cost_function(graph.neighborhood(v.id))
        if cost > max_cost:
            max_cost = cost

    return max_cost


if __name__ == "__main__":

    # Carlone SDP solution.
    vertices, edges = parse_g2o("../../../../datasets/input_MITb_g2o.g2o")
    # positions, rotations, _ = pgo(vertices, edges)
    #
    # for i, v in enumerate(vertices):
    #     v.set_state(positions[i], rotations[i])

    # Generate pose graph matrix.
    graph = Graph(vertices, edges)

    rng = np.random.SFC64()

    plot_vertices(graph.vertices)

    i = 0
    changes = 0
    max_costs = []

    while changes < 1000:

        #i = rng.random_raw() % (500)

        i = (i + 1) % 500

        pre_cost = cost_function(graph.neighborhood(i))

        if pre_cost < 1:
            continue

        old_position = vector_to_complex(graph.vertices[i].position)
        old_rotation = vector_to_complex(rotation_vector(graph.vertices[i].rotation))[0]

        changes += 1
        print("update: \t\t\t%d \ni: \t\t\t\t\t%d" % (changes, i))

        position, rotation = solve_local_sdp(i, graph, verbose=False)

        # Plot original neighborhood.
        #plot_vertices(graph.neighborhood(i).vertices, color='b', edges=graph.neighborhood(i).edges, labels=True)

        # Update graph with solution.
        graph.set_state(i, position, rotation)

        print("Old cost: \t\t\t%f\nNew cost: \t\t\t%f\nPercent change: \t%f" % (pre_cost, cost_function(graph.neighborhood(i)), (pre_cost - cost_function(graph.neighborhood(i))) / pre_cost))

        # Plot updated neighborhood.
        #plot_vertices(graph.neighborhood(i).vertices, new_figure=True, color='r', edges=graph.neighborhood(i).edges, labels=True)
        print("\n")


    plot_vertices(graph.vertices, labels=True)
    draw_plots()