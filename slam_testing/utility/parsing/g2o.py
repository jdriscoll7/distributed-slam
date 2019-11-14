import numpy as np


class Vertex:

    def __init__(self, id, state):
        self.id = id
        self.state = state


class Edge:

    def __init__(self, out_vertex, in_vertex, relative_pose, rotation, information_matrix):
        self.out_vertex = out_vertex
        self.in_vertex = in_vertex
        self.relative_pose = relative_pose
        self.rotation = rotation
        self.information_matrix = information_matrix


def parse_g2o(path):
    """
    :param path: Path to g2o file.
    :return:     List of vertices and list of edges.
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
            relative_pose = np.array([float(x) for x in line_tokens[3:5]]).T
            rotation = float(line_tokens[5])
            matrix = np.matrix([[float(line_tokens[6]), float(line_tokens[7]), float(line_tokens[8])],
                               [float(line_tokens[7]), float(line_tokens[9]), float(line_tokens[10])],
                               [float(line_tokens[8]), float(line_tokens[10]), float(line_tokens[11])]])

            # Update return list with information just extracted.
            edges.insert(out_vertex, Edge(out_vertex, in_vertex, relative_pose, rotation, matrix))

    return vertices, edges


def write_g2o(vertices, edges, file_name):

    # Open data file.
    file = open(file_name, 'w')

    # Write lines for vertices.
    for vertex in vertices:

        # Convert state into a string with space delimiters.
        state_string = " ".join(map(str, [x[0] for x in vertex.state]))

        # Write line to file.
        file.write("VERTEX_SE2 " + str(vertex.id) + " " + state_string + "\n")

    for edge in edges:

        # Get relative pose as string.
        relative_pose = " ".join(map(str, edge.relative_pose[0])) + " " + " ".join(map(str, edge.relative_pose[1]))

        # Create tokens before info matrix.
        edge_string = " ".join([str(edge.out_vertex), str(edge.in_vertex), relative_pose, str(edge.rotation)])

        # Convert information matrix into string with space delimiters.
        matrix_string = " ".join(str(x[1]) for x in np.ndenumerate(edge.information_matrix[np.triu_indices(3)]))

        # Write line to file.
        file.write("EDGE_SE2 " + edge_string + " " + matrix_string + "\n")

    # Close data file.
    file.close()
