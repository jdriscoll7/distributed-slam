import numpy as np
from matplotlib import pyplot as plt


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
            vertices.insert(vertex_id, vertex_state)

        else:

            # Extract ID and information matrix.
            out_id = int(line_tokens[1])
            in_id = int(line_tokens[2])
            relative_pose = np.array([line_tokens[3:5]]).T
            matrix = np.array([[float(line_tokens[6]), float(line_tokens[7]), float(line_tokens[8])],
                               [float(line_tokens[7]), float(line_tokens[9]), float(line_tokens[10])],
                               [float(line_tokens[8]), float(line_tokens[10]), float(line_tokens[11])]])

            # Update return list with information just extracted.
            #edges.insert(edge_id, matrix)

    # Convert vertices to numpy array.
    vertices = np.asarray(vertices).reshape((-1, 3))

    return vertices, edges


def plot_g2o_vertices(vertices):

    plt.figure()
    plt.plot(vertices[:, 0], vertices[:, 1], '-')
    plt.axis('off')


if __name__ == "__main__":
    vertices, edges = parse_g2o("../datasets/input_INTEL_g2o.g2o")
    print(len(vertices[:, 0]))
    print(len(vertices[:, 1]))