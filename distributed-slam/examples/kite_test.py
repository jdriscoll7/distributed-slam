import numpy as np

from solvers.experimental.fixed_sdp import rotation_matrix
from solvers.experimental.local_admm import cost_function
from solvers.sdp import pgo
from utility.graph import Vertex, Edge, Graph
from utility.visualization import plot_pose_graph


def kite(n, d, ladder_length=3, position_sigma=0, rotation_sigma=0):
    """
    Creates kite graph with n vertices, as well as the kite with ladder (2n - d + 3) vertices).

    :param n: Number of vertices in handle-free kite.
    :return:  (kite graph, kite with handle graph)
    """

    np.random.seed(123123)

    # Graph is not well-defined or useful for n less than or equal to 4.
    assert n > 4

    # List of vertices and edges for kite graph.
    vertices = []
    edges = []

    # The first four vertices (and the edges between them) are always the same.
    for i, (p, r) in enumerate([([0, 1], 0), ([1, 0], 0), ([-1, 0], 0), ([0, -1], -np.pi / 2)]):
        vertices.append(Vertex(position=np.asarray(p), rotation=r, id=i))

    for (i, j) in [(0, 1), (1, 2), (2, 3), (2, 0), (3, 0), (3, 1)]:
        relative_pose = rotation_matrix(vertices[i].rotation).T @ (vertices[j].position - vertices[i].position)
        relative_rotation = vertices[j].rotation - vertices[i].rotation
        edges.append(Edge(out_vertex=i,
                          in_vertex=j,
                          relative_pose=relative_pose,
                          rotation=relative_rotation,
                          information_matrix=None))

    # Add remaining vertices and edges in kite.
    for i in range(4, n):
        vertices.append(Vertex(position=np.asarray([0, 2 - i]), rotation=-np.pi / 2, id=i))

        relative_pose = rotation_matrix(vertices[i - 1].rotation).T @ (vertices[i].position - vertices[i - 1].position)
        relative_rotation = vertices[i].rotation - vertices[i - 1].rotation

        edges.append(Edge(out_vertex=i - 1,
                          in_vertex=i,
                          relative_pose=relative_pose,
                          rotation=relative_rotation,
                          information_matrix=None))

    # Kite without ladder - save now to return later.
    kite_no_handle = Graph(vertices, edges)

    # Add rungs of ladder to kite.
    rung_offset = [1, 0]
    for i in range(d + 3, d + 3 + ladder_length):
        vertices.append(Vertex(position=vertices[i].position + np.asarray(rung_offset), rotation=0, id=n + i - (d + 3)))

        relative_pose = rotation_matrix(vertices[n + i - (d + 3)].rotation).T @ (vertices[i].position - vertices[n + i - (d + 3)].position)
        relative_rotation = vertices[i].rotation - vertices[n + i - (d + 3)].rotation

        edges.append(Edge(out_vertex=n + i - (d + 3),
                          in_vertex=i,
                          relative_pose=relative_pose + position_sigma*np.random.randn(*relative_pose.shape),
                          rotation=relative_rotation + rotation_sigma*np.random.randn(),
                          information_matrix=None))

    # Connect rungs.
    for i in range(n, n + ladder_length - 1):

        relative_pose = rotation_matrix(vertices[i + 1].rotation).T @ (vertices[i].position - vertices[i + 1].position)
        relative_rotation = vertices[i].rotation - vertices[i + 1].rotation

        edges.append(Edge(out_vertex=i+1,
                          in_vertex=i,
                          relative_pose=relative_pose + position_sigma * np.random.randn(*relative_pose.shape),
                          rotation=relative_rotation + rotation_sigma * np.random.randn(),
                          information_matrix=None))

    return kite_no_handle, Graph(vertices, edges)


if __name__ == "__main__":

    solutions = []

    position_sigma = 5
    rotation_sigma = 1

    no_handle_kite = kite(n=30, d=10, position_sigma=position_sigma, rotation_sigma=rotation_sigma)[0]

    # Solve global problem and store positions and rotations.
    positions, rotations = pgo(graph=no_handle_kite)[0:2]

    no_handle_kite.update_states([i for i in range(len(no_handle_kite.vertices))], np.vstack((positions, rotations)))
    print(cost_function(no_handle_kite))

    for i in range(5, 20, 1):

        # Create kite with handle.
        handle_kite = kite(n=30, d=i, position_sigma=position_sigma, rotation_sigma=rotation_sigma)[1]

        # Solve global problem and store positions and rotations.
        positions, rotations = pgo(graph=handle_kite)[0:2]

        handle_kite.update_states([i for i in range(len(handle_kite.vertices))], np.vstack((positions, rotations)))
        print(cost_function(handle_kite))
        solutions.append(handle_kite)

    print(solutions)


