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
from utility.visualization import plot_complex_list, draw_plots, plot_pose_graph, plot_line_graph
from utility.parsing import parse_g2o
from utility.data_generation import random_data
import matplotlib.gridspec as gridspec
from solvers.admm.local_admm import cost_function


def changes_from_edge(path):

    # Parse data set twice - one contains one more edge than the other.
    large_graph = Graph(*parse_g2o(path))
    small_graph = large_graph.copy()

    # Edge to remove.
    removal_edge = find_off_path_edge(large_graph)

    # Remove one edge from the smaller graph.
    small_graph.remove_edges([removal_edge])

    # Find distance of all poses from the critical edge.
    distances = graphical_distances(graph=large_graph, edges=[removal_edge])

    print("Edge removed: (%d, %d)" % (removal_edge.out_vertex + 1, removal_edge.in_vertex + 1))

    # Solve for both graphs.
    large_solution, large_dual = pgo(large_graph)
    small_solution, small_dual = pgo(small_graph)

    # Print costs of each solution.
    print("Cost of large solution: " + str(cost_function(large_solution)))
    print("Cost of small solution: " + str(cost_function(small_solution)))

    # Store magnitude of changes in dictionary with vertex id as key.
    position_changes = {}
    rotation_changes = {}

    # Find the magnitude of change for each pose.
    for v in large_graph.vertices:
        position_change = np.linalg.norm(v.position - small_solution.get_vertex(v.id).position)
        rotation_change = np.mod(np.abs(v.rotation - small_solution.get_vertex(v.id).rotation), np.pi)

        position_changes[v.id] = position_change + rotation_change
        rotation_changes[v.id] = rotation_change

    # Create x and y data points for distance vs. magnitude change graph.
    x_data = []
    position_data = []
    rotation_data = []

    for key in position_changes:
        x_data.append(distances[key])
        position_data.append(position_changes[key])
        rotation_data.append(rotation_changes[key])

    plt.figure()
    plt.scatter(x_data, position_data)
    plt.title("Position Change vs. Distance")

    plt.figure()
    plt.scatter(x_data, rotation_data)
    plt.title("Rotation Change vs. Distance")

    plt.figure()
    plot_pose_graph(graph=large_solution)

    plt.figure()
    plot_pose_graph(graph=small_solution)

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
    # FILE_PATH = "datasets/good_geometric_example.g2o"

    FILE_PATH = "datasets/CSAIL_P.g2o"

    changes_from_edge(FILE_PATH)
