import numpy as np
from utility.parsing import Vertex, Edge, write_g2o
from .grouped_data import group_datasets


def _rotation_matrix(theta):

    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def _angle_from_rotation_matrix(matrix):

    return np.arctan2(matrix[1, 0], matrix[0, 0])


def _create_edge_list(edge_ids, translation_measurements, rotation_measurements):

    # List comprehension to turn inputs into list of Edge objects.
    return [Edge(edge_ids[i][0], edge_ids[i][1], translation_measurements[i],
                 _angle_from_rotation_matrix(rotation_measurements[i]), np.eye(3))
            for i in range(len(edge_ids))]


def _create_vertex_list(truth_states):

    # List comprehension to turn input into list of Vertex objects.
    return [Vertex(i, position=truth_states[i][0:2], rotation=truth_states[i][2]) for i in range(len(truth_states))]


def geometric_graph_edges(vertex_coordinates, radius):

    # Keep list of edges included in geometric graph.
    edge_set = []

    # Find largest distance between any two nodes.
    closest_vertex_distance = [[np.inf, np.inf] for _ in range(len(vertex_coordinates))]
    for i, vertex_1 in enumerate(vertex_coordinates):
        for j, vertex_2 in enumerate(vertex_coordinates):
            if i != j:
                distance = np.linalg.norm(vertex_1 - vertex_2)

                if distance < closest_vertex_distance[i][0]:
                    closest_vertex_distance[i][1] = closest_vertex_distance[i][0]
                    closest_vertex_distance[i][0] = distance
                elif distance < closest_vertex_distance[i][1]:
                    closest_vertex_distance[i][1] = distance

    if max([max(x) for x in closest_vertex_distance]) > radius:
        radius = max([max(x) for x in closest_vertex_distance]) + 0.1
        print("Given radius too small. Radius expanded to %f" % (radius))

    for i, vertex_1 in enumerate(vertex_coordinates):
        for j, vertex_2 in enumerate(vertex_coordinates):
            # Keep counter to alternate ingoing/outgoing edges.
            edge_counter = 0
            if i < j and np.linalg.norm(vertex_1 - vertex_2) <= radius:
                if edge_counter % 2 == 0:
                    edge_set.append((i, j))
                else:
                    edge_set.append((j, i))
                edge_counter += 1

    return edge_set


def create_geometric_dataset(translation_variance, rotation_variance, n_poses, box=(0, 10, 0, 10), radius=1, file_name=None):

    # Generate truth poses and rotations.
    truth_coordinates = [np.asarray([np.random.uniform(box[0], box[1]), np.random.uniform(box[2], box[3])]).reshape((2, 1)) for _ in range(n_poses)]
    truth_rotations = [_rotation_matrix(np.random.uniform(-np.pi, np.pi)) for i in range(n_poses)]

    # Create spanning cycle that traverses vertices in order.
    edge_ids = geometric_graph_edges(vertex_coordinates=truth_coordinates, radius=radius)

    # Generate relative pose measurements based on model.
    translation_measurements = [truth_rotations[edge[0]].T @ (truth_coordinates[edge[1]] - truth_coordinates[edge[0]])
                                + np.random.multivariate_normal(np.zeros((2,)), translation_variance * np.eye(2)).reshape(-1, 1)
                                for edge in edge_ids]

    # Generate relative rotation measurements based on model.
    rotation_measurements = [truth_rotations[edge[0]].T @ truth_rotations[edge[1]]
                             @ _rotation_matrix(np.random.normal(0, rotation_variance)) for edge in edge_ids]

    # Stack coordinates and rotations for states.
    truth_states = [np.vstack([truth_coordinates[i], _angle_from_rotation_matrix(truth_rotations[i])])
                    for i in range(n_poses)]

    # Turn everything into vertices and edges.
    vertices = _create_vertex_list(truth_states)
    edges = _create_edge_list(edge_ids, translation_measurements, rotation_measurements)

    # Write results if desired.
    if file_name != None:
        write_g2o(vertices, edges, file_name)

    return vertices, edges


if __name__ == "__main__":

    create_geometric_dataset(2, 3, 10, 'test.g2o')
