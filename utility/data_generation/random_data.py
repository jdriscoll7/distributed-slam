import numpy as np
from utility.parsing.g2o import Vertex, Edge, write_g2o


def rotation_matrix(theta):

    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def angle_from_rotation_matrix(matrix):

    return np.arctan2(matrix[1, 0], matrix[0, 0])


def create_random_dataset(translation_variance, rotation_variance, n_poses, file_name):

    # Generate truth poses and rotations.
    truth_coordinates = [np.random.uniform(0, 10, (2, 1)) for i in range(0, n_poses)]
    truth_rotations = [rotation_matrix(np.random.uniform(-np.pi, np.pi)) for i in range(0, n_poses)]

    # Create spanning path.
    path = np.random.permutation([i for i in range(0, n_poses)])
    edge_ids = [(path[i - 1], path[i]) for i in range(1, n_poses)]

    # Generate relative pose measurements based on model.
    translation_measurements = [truth_rotations[edge[0]].T @ (truth_coordinates[edge[1]] - truth_coordinates[edge[0]])
                                + np.random.normal(np.zeros((2, 1)), translation_variance * np.eye(2)) for edge in edge_ids]

    # Generate relative rotation measurements based on model.
    rotation_measurements = [truth_rotations[edge[0]].T @ truth_rotations[edge[1]]
                             @ rotation_matrix(np.random.normal(0, rotation_variance)) for edge in edge_ids]

    # Stack coordinates and rotations for states.
    truth_states = [np.vstack([truth_coordinates[i], angle_from_rotation_matrix(truth_rotations[i])]) for i in range(0, n_poses)]

    # Turn everything into vertices and edges.
    vertices = [Vertex(i, truth_states[i]) for i in range(0, n_poses)]
    edges = [Edge(edge_ids[i][0], edge_ids[i][1], translation_measurements[i], angle_from_rotation_matrix(rotation_measurements[i]),
                  np.eye(3))
             for i in range(0, len(edge_ids))]

    # Write results.
    write_g2o(vertices, edges, file_name)


if __name__ == "__main__":

    create_random_dataset(2, 3, 10, 'testttttttttttt.g2o')