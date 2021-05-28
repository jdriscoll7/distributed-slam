import numpy as np
from os import makedirs
from os.path import dirname

from utility.graph import *


def parse_g2o(path, groups=False):
    """
    :param path:   Path to g2o file.
    :param groups: True if vertices are expected to have group tags.
    :return:       List of vertices and list of edges.
    """

    # Initialize return lists.
    edges = []
    vertices = []

    # Parse file line-by-line based on g2o format.
    file = open(path)
    for line in file:

        # Tokens are separated by spaces.
        line_tokens = line.split(' ')

        # Line is either vertex or edge. Process accordingly.
        if "VERTEX" in line_tokens[0]:

            # Extract ID and state (position) vector.
            vertex_id = int(line_tokens[1])
            vertex_position = np.array([float(i) for i in line_tokens[2:4]]).T
            vertex_rotation = float(line_tokens[4])

            # Update return list with information just extracted (force state into column vector).
            if groups:

                # Extract number that comes after string in last token of form "GROUP#".
                group_id = int(line_tokens[5][5:])

                # Adjust number of groups in "vertices" list of lists if group number is too high.
                if len(vertices) <= group_id:
                    vertices += [[] for i in range(len(vertices) - group_id + 1)]

                # Append vertex to list according to its group.
                vertices[group_id].append(Vertex(vertex_id, vertex_position, vertex_rotation))

            else:
                vertices.append(Vertex(vertex_id, vertex_position, vertex_rotation))

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

    # Return different types based on if groups are used.
    if groups:
        return MultiGraph(vertices, edges)
    else:
        return vertices, edges


def write_g2o(vertices, edges, file_name, group_ids=None):

    # Make sure directory of file name exists - if not, create it.
    if dirname(file_name) is not '':
        makedirs(dirname(file_name), exist_ok=True)

    # Open data file.
    file = open(file_name, 'w')

    # Write lines for vertices.
    for vertex in vertices:

        # Convert state into a string with space delimiters.
        state_string = " ".join(map(str, [x[0] for x in np.vstack((vertex.position, vertex.rotation))]))

        # Write line to file.
        file.write("VERTEX_SE2 " + str(vertex.id) + " " + state_string)

        # Optional group tag.
        if group_ids != None:
            file.write(" GROUP" + str(group_ids[vertex.id]))

        # Need to make new line in either case.
        file.write("\n")

    for edge in edges:

        # Get relative pose as string.
        relative_pose = " ".join(map(str, [edge.relative_pose[0][0], edge.relative_pose[1][0]]))

        # Create tokens before info matrix.
        edge_string = " ".join([str(edge.out_vertex), str(edge.in_vertex), relative_pose, str(edge.rotation)])

        # Convert information matrix into string with space delimiters.
        matrix_string = " ".join(str(x[1]) for x in np.ndenumerate(edge.information_matrix[np.triu_indices(3)]))

        # Write line to file.
        file.write("EDGE_SE2 " + edge_string + " " + matrix_string + "\n")

    # Close data file.
    file.close()
