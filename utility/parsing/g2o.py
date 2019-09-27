import numpy as np


class Vertex:

    def __init__(self, number, state):
        self.number = number
        self.state = state


class Edge:

    def __init__(self, out_vertex, in_vertex, relative_pose, information_matrix):
        self.out_vertex = out_vertex
        self.in_vertex = in_vertex
        self.relative_pose = relative_pose
        self.information_matrix = information_matrix


def parse_g2o(path):
    """
    :param path: Path to g2o file.
    :return: List of vertices and list of edges.
    """

    # Initialize return lists.
    vertices, edges = [], []

    # Parse file line-by-line based on g2o format.
    file = open(path)
    for line in file:

        # Tokens are separated by spaces.
        line_tokens = line.split(' ')

        # Line is either vertex or edge. Process accordingly.
        if "VERTEX" in line_tokens[0]:

            # Extract ID and state (position) vector.
            vertex_id = int(line_tokens[1])
            vertex_state = np.array([float(i) for i in line_tokens[2:]]).T

            # Update return list with information just extracted (force state into column vector).
            vertices.insert(vertex_id, Vertex(vertex_id, vertex_state))

        else:

            # Extract ID and information matrix.
            out_vertex = int(line_tokens[1])
            in_vertex = int(line_tokens[2])
            relative_pose = np.array([line_tokens[3:5]]).T
            matrix = np.array([[float(line_tokens[6]), float(line_tokens[7]), float(line_tokens[8])],
                               [float(line_tokens[7]), float(line_tokens[9]), float(line_tokens[10])],
                               [float(line_tokens[8]), float(line_tokens[10]), float(line_tokens[11])]])

            # Update return list with information just extracted.
            edges.insert(out_vertex, Edge(out_vertex, in_vertex, relative_pose, matrix))

    return vertices, edges