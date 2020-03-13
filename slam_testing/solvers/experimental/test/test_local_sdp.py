import numpy as np

from solvers.experimental.fixed_sdp import solve_local_sdp, rotation_vector, vector_to_complex
from utility.graph import Graph
from utility.parsing import parse_g2o
from utility.visualization import plot_vertices, draw_plots


def averaging(vertex, graph):

    vertex_mapping = dict([(v.id, i) for i, v in enumerate(graph.vertices)])

    in_edges = [e for e in graph.edges if e.in_vertex == vertex]
    out_edges = [e for e in graph.edges if e.out_vertex == vertex]

    in_relative_poses = [graph.vertices[vertex_mapping[e.in_vertex]].position.reshape(-1, 1)
                       + e.relative_pose.reshape(-1, 1)
                         for e in in_edges]

    out_relative_poses = [graph.vertices[vertex_mapping[e.out_vertex]].position.reshape(-1, 1)
                        + e.relative_pose.reshape(-1, 1)
                         for e in out_edges]

    in_rotations = [e.rotation for e in in_edges]
    out_rotations = [-e.rotation for e in out_edges]

    position = np.mean(in_relative_poses + out_relative_poses, axis=0)
    rotation = np.mean(in_rotations + out_rotations)

    return position, rotation


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

    for _ in range(1, len(vertices) - 1):

        i = rng.random_raw() % (len(vertices) - 1)

        old_position = graph.vertices[i].position
        old_rotation = vector_to_complex(rotation_vector(graph.vertices[i].rotation))

        #position, rotation = solve_local_sdp(i, graph)
        position, rotation = averaging(i, graph)

        # Plot original neighborhood.
        #plot_vertices(graph.neighborhood(i).vertices, color='b', edges=graph.neighborhood(i).edges)

        # Update graph with solution.
        graph.set_state(i, position, rotation)

        # Plot updated neighborhood.
        #plot_vertices(graph.neighborhood(i).vertices, new_figure=True, color='r', edges=graph.neighborhood(i).edges)

        print(i)


    plot_vertices(graph.vertices)
    draw_plots()