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


def form_graph_sequence(graph, pose_id, step=1):

    # Form list of vertices in order of distance from fixed pose.
    distance_dictionary = {}

    for v in graph.vertices:
        distance_dictionary[v.id] = np.linalg.norm(graph.get_vertex(pose_id).position - v.position)

    sorted_vertex_list = sorted(distance_dictionary, key=distance_dictionary.get)

    # List of graphs to run optimization on.
    graph_list = []
    vertex_id_list = []

    for i in range(1, len(sorted_vertex_list) // step):
        subgraph, vertex_ids = graph.subgraph(i=sorted_vertex_list[:i*step], reduce=True)
        graph_list.append(subgraph)
        vertex_id_list.append(vertex_ids)

    return graph_list, vertex_id_list, sorted_vertex_list


def sequential_fixed_pose(file_path, vertex_id, step=1):

    # Parse data set twice - one contains one more edge than the other.
    main_graph = Graph(*parse_g2o(file_path))
    n_vertices = len(main_graph.vertices)

    # Solve for large graph and print cost of solution.
    main_solution, large_dual = pgo(main_graph)
    print("Cost of large solution: " + str(cost_function(main_solution)))

    # Keep track of magnitude changes based on distance.
    change_magnitudes = {}
    rotation_changes = {}

    # Create list of graphs to perform optimization on.
    graph_list, vertex_id_list, vertex_order = form_graph_sequence(graph=main_graph, pose_id=vertex_id, step=step)

    # Loop through all edges in graph, removing just that edge from original graph and resolving.
    for i, graph in enumerate(graph_list[1:]):

        changed_vertex_id = vertex_id_list[i + 1][vertex_id]
        added_vertex_id = vertex_id_list[i + 1][vertex_order[i + 1]]

        # Find edges attached to focused vertex.
        edge_neighborhood = [e for e in graph.edges if e.in_vertex == changed_vertex_id or e.out_vertex == changed_vertex_id]

        # Find distance of target pose to measurement.
        pose_distance = graphical_distances(graph=graph, edges=edge_neighborhood)[added_vertex_id]

        # Solve for new graph.
        solution, dual = pgo(graph)

        solution_angle = np.angle(solution.vertices[1].position[0] + 1j * solution.vertices[1].position[1])
        combined_angle = np.angle(main_solution.vertices[1].position[0] + 1j * main_solution.vertices[1].position[1])
        solution.rotate(combined_angle - solution_angle)

        plt.figure()
        plot_pose_graph(graph=solution)

        # Print costs of new solution.
        print("Cost of perturbed solution %d: %f" % (i, cost_function(solution)))

        # Find the magnitude of change for selected vertex.
        target_vertex = solution.get_vertex(changed_vertex_id)

        position_change = np.linalg.norm(target_vertex.position - main_solution.get_vertex(changed_vertex_id).position)
        rotation_change = np.mod(np.abs(target_vertex.rotation - main_solution.get_vertex(changed_vertex_id).rotation), np.pi)

        # Record the magnitude of change for target pose for removed edge.
        change_magnitudes.setdefault(pose_distance, []).append(position_change)
        rotation_changes.setdefault(pose_distance, []).append(rotation_change)
        print("Distance between edge and pose: " + str(pose_distance))
        print("Position change caused by edge removal: " + str(position_change))
        print("Rotation change caused by edge removal: " + str(rotation_change))

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
    plot_pose_graph(graph=main_solution)

    draw_plots()


if __name__ == "__main__":
    # create_random_dataset(translation_variance=0.2,
    #                       rotation_variance=0.2,
    #                       n_poses=100,
    #                       edge_probability=0.005,
    #                       file_name="datasets/random_test.g2o")

    # Path for generated file.
    # FILE_PATH = "datasets/random_test.g2o"

    create_geometric_dataset(translation_variance=0.2,
                             rotation_variance=0.2,
                             n_poses=10,
                             box=(0, 10, 0, 10),
                             radius=1,
                             file_name="datasets/random_geometric.g2o")

    # Path for generated file.
    FILE_PATH = "datasets/random_geometric.g2o"

    # graph_list = form_graph_sequence(Graph(*parse_g2o("datasets/good_geometric_example.g2o")), pose_id=3)

    # for graph in graph_list[2:10]:
    #     plt.figure()
    #     plot_pose_graph(graph)
    #
    # draw_plots()

    sequential_fixed_pose(FILE_PATH, vertex_id=3, step=1)
