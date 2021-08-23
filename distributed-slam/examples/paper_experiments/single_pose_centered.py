import copy
import random

from helper_functions import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from os import makedirs
from os.path import dirname
from random import shuffle
import pickle

from solvers.sdp import pgo
from utility.data_generation import serial_graph_plan
from utility.data_generation import create_random_dataset
from utility.data_generation.geometric_graphs import create_geometric_dataset
from utility.graph import Graph, Edge
from utility.visualization import plot_complex_list, draw_plots, plot_pose_graph
from utility.parsing import parse_g2o
from utility.data_generation import random_data
import matplotlib.gridspec as gridspec
from solvers.admm.local_admm import cost_function


def changes_for_fixed_pose(file_path, vertex_id, n_edges=10):

    # Parse data set twice - one contains one more edge than the other.
    large_graph = Graph(*parse_g2o(file_path))
    n_vertices = len(large_graph.vertices)

    # Solve for large graph and print cost of solution.
    large_solution, large_dual = pgo(large_graph)
    print("Cost of large solution: " + str(cost_function(large_solution)))

    # Obtain all measurements in graph - shuffle list for random removal later.
    edges = large_graph.get_edges()
    shuffle(edges)

    # Keep track of magnitude changes based on distance.
    change_magnitudes = {}
    rotation_changes = {}

    # Loop through all edges in graph, removing just that edge from original graph and resolving.
    edges_removed = 0
    for edge in edges:

        # Copy large graph for each edge.
        small_graph = large_graph.copy()

        # Make sure graph stays connected by avoiding main path.
        if edge_not_in_path(edge, n_vertices):

            # Keep track of number of removed edges.
            edges_removed += 1

            # Remove selected edge.
            small_graph.remove_edges([edge])

            # Find distance of target pose to measurement.
            pose_distance = graphical_distances(graph=large_graph, edges=[edge])[vertex_id]

            print("Edge removed: (%d, %d)" % (edge.out_vertex + 1, edge.in_vertex + 1))

            # Solve for new graph.
            small_solution, small_dual = pgo(small_graph)

            # Print costs of new solution.
            print("Cost of perturbed solution: " + str(cost_function(small_solution)))

            # Find the magnitude of change for selected vertex.
            target_vertex = small_solution.get_vertex(vertex_id)

            position_change = np.linalg.norm(target_vertex.position - large_solution.get_vertex(vertex_id).position)
            rotation_change = np.mod(np.abs(target_vertex.rotation - large_solution.get_vertex(vertex_id).rotation), np.pi)

            # Record the magnitude of change for target pose for removed edge.
            change_magnitudes.setdefault(pose_distance, []).append(position_change)
            rotation_changes.setdefault(pose_distance, []).append(rotation_change)
            print("Distance between edge and pose: " + str(pose_distance))
            print("Position change caused by edge removal: " + str(position_change))
            print("Rotation change caused by edge removal: " + str(rotation_change))

        if edges_removed == n_edges:
            break

    # Create x and y data points for distance vs. magnitude change graph.
    position_x_data = []
    rotation_x_data = []
    average_x_data = []
    position_data = []
    rotation_data = []
    average_data = []
    average_log_data = []
    average_rotation_data = []

    for key in change_magnitudes:
        average_data.append(np.mean(change_magnitudes[key]))
        average_log_data.append(np.mean(np.log10(change_magnitudes[key])))
        average_rotation_data.append(np.mean(rotation_changes[key]))
        average_x_data.append(key)
        for magnitude in change_magnitudes[key]:
            position_x_data.append(key)
            position_data.append(magnitude)
        for rotation_magnitude in rotation_changes[key]:
            rotation_x_data.append(key)
            rotation_data.append(rotation_magnitude)

    # Save data.
    with open("plot_data.pickle", "wb") as f:
        pickle.dump([position_x_data, rotation_x_data, average_x_data, position_data, rotation_data, average_data, average_rotation_data], f)

    plt.figure()
    plt.scatter(position_x_data, position_data)
    plt.title("Magnitude of Change vs. Distance")
    plt.xlabel("Distance from Pose to Added Edge")
    plt.ylabel("Magnitude of Change in Pose")

    plt.figure()
    plt.scatter(position_x_data, np.log10(position_data))
    plt.title("Log-Magnitude of Change vs. Distance")
    plt.xlabel("Distance from Pose to Added Edge")
    plt.ylabel("Magnitude of Change in Pose")

    plt.figure()
    plt.scatter(average_x_data, average_log_data)
    plt.title("Average Log-Magnitude of Change vs. Distance")
    plt.xlabel("Distance from Pose to Added Edge")
    plt.ylabel("Magnitude of Change in Pose")

    plt.figure()
    plt.scatter(rotation_x_data, rotation_data)
    plt.title("Magnitude of Rotation Change vs. Distance")
    plt.xlabel("Distance from Pose to Added Edge")
    plt.ylabel("Magnitude of Change in Rotation")

    plt.figure()
    plt.scatter(average_x_data, average_data)
    plt.title("Average Magnitude of Change vs. Distance")
    plt.xlabel("Distance from Pose to Added Edge")
    plt.ylabel("Magnitude of Change in Pose")

    plt.figure()
    plt.scatter(average_x_data, average_rotation_data)
    plt.title("Average Magnitude of Rotation Change vs. Distance")
    plt.xlabel("Distance from Pose to Added Edge")
    plt.ylabel("Magnitude of Change in Rotation")

    plt.figure()
    plot_pose_graph(graph=large_solution)

    draw_plots()


if __name__ == "__main__":
    # create_random_dataset(translation_variance=0.2,
    #                       rotation_variance=0.2,
    #                       n_poses=100,
    #                       edge_probability=0.005,
    #                       file_name="datasets/random_test.g2o")

    # Path for generated file.
    # FILE_PATH = "datasets/random_test.g2o"

    # create_geometric_dataset(translation_variance=0.2,
    #                          rotation_variance=0.2,
    #                          n_poses=40,
    #                          box=(0, 10),
    #                          radius=1,
    #                          file_name="datasets/random_geometric.g2o")

    # Path for generated file.
    FILE_PATH = "datasets/good_geometric_example.g2o"

    changes_for_fixed_pose(FILE_PATH, vertex_id=17, n_edges=15)
