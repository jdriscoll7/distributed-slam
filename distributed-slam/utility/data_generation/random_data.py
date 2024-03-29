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


def create_measurement(graph, i, j, translation_variance, rotation_variance):

    # Truth states.
    truth_coordinates = [np.reshape(graph.get_vertex(i).position, (2, 1)), np.reshape(graph.get_vertex(j).position, (2, 1))]
    truth_rotations = [_rotation_matrix(graph.get_vertex(i).rotation).reshape((2, 2)), _rotation_matrix(graph.get_vertex(j).rotation).reshape((2, 2))]

    # Generate relative pose measurements based on model.
    translation_measurement = truth_rotations[0].T @ (truth_coordinates[1] - truth_coordinates[0]) \
                              + np.random.multivariate_normal(np.zeros((2,)), translation_variance * np.eye(2)).reshape(-1, 1)

    # Generate relative rotation measurements based on model.
    rotation_measurement = truth_rotations[0].T @ truth_rotations[1] @ _rotation_matrix(np.random.normal(0, rotation_variance))

    # Reshape translation measurement to be safe.
    translation_measurement = np.reshape(translation_measurement, (2, 1))

    # Turn everything into vertices and edges.
    edge = _create_edge_list([(i, j)], [translation_measurement], [rotation_measurement])[0]

    return edge


def create_random_dataset(translation_variance, rotation_variance, n_poses, edge_probability=0.1, file_name=None, box=[0, 10, 0, 10]):

    # Generate truth poses and rotations.
    truth_coordinates = [np.asarray([np.random.uniform(box[0], box[1]), np.random.uniform(box[2], box[3])]).reshape((2, 1)) for _ in range(n_poses)]
    truth_rotations = [_rotation_matrix(np.random.uniform(-np.pi, np.pi)) for i in range(n_poses)]

    # Create spanning cycle that traverses vertices in order.
    edge_ids = [(i - 1, i) for i in range(1, n_poses)]
    #edge_ids.append((n_poses - 1, 0))

    # Randomly add in cross edges. Exclude already added edges.
    # Sample uniform RV to determine if edge should be added.
    for i in range(n_poses):
        for j in range(n_poses):
            if abs(i - j) > 1 and np.random.uniform(0, 1) > 1 - edge_probability:
                edge_ids.append((i, j))

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


def create_random_grouped_dataset(translation_variance, rotation_variance, n_poses, file_name):

    vertex_lists, edge_lists = [], []

    for n in n_poses:

        # Create random dataset for one group.
        v, e = create_random_dataset(translation_variance, rotation_variance, n)

        # Append created graph to lists.
        vertex_lists.append(v)
        edge_lists.append(e)

    multigraph = group_datasets(vertex_lists, edge_lists)
    vertices, edges, group_ids = multigraph.get_full_graph()

    write_g2o(vertices, edges, file_name, group_ids=group_ids)

    return multigraph


if __name__ == "__main__":

    create_random_dataset(2, 3, 10, 'testttttttttttt.g2o')
