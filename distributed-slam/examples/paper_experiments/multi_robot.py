import copy
import random

import networkx as nx

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
from utility.data_generation import create_random_dataset, create_measurement
from utility.data_generation.geometric_graphs import create_geometric_dataset
from utility.graph import Graph, Edge
from utility.visualization import plot_complex_list, draw_plots, plot_pose_graph, plot_line_graph
from utility.parsing import parse_g2o, write_g2o
from utility.data_generation import random_data
import matplotlib.gridspec as gridspec
from solvers.admm.local_admm import cost_function


def shift_graph(graph, shift_vector):

    shift_vector = np.reshape(shift_vector, graph.vertices[0].position.shape)

    for i, v in enumerate(graph.vertices):
        graph.vertices[i].position = v.position + shift_vector

    return graph


def create_combined_dataset(graph_1, graph_2, n_cross_edges, translation_variance=0.2, rotation_variance=0.2, combined_path=None):

    if combined_path is None:
        combined_path = "datasets/combined_dataset.g2o"

    # Offset id's in edges and vertices of second graph.
    for i, v in enumerate(graph_2.get_vertices()):
        graph_2.vertices[i].id = v.id + len(graph_1.vertices)

    for i, e in enumerate(graph_2.get_edges()):
        graph_2.edges[i].in_vertex = e.in_vertex + len(graph_1.vertices)
        graph_2.edges[i].out_vertex = e.out_vertex + len(graph_1.vertices)

    # Form new graph and save data.
    combined_vertices = graph_1.get_vertices() + graph_2.get_vertices()
    combined_edges = graph_1.get_edges() + graph_2.get_edges()
    combined_graph = Graph(vertices=combined_vertices, edges=combined_edges)

    # Evaluate distance between all pairs of vertices from different graphs.
    distance_dictionary = {}
    for v_1 in graph_1.get_vertices():
        for v_2 in graph_2.get_vertices():
            distance_dictionary[(v_1.id, v_2.id)] = np.linalg.norm(v_1.position - v_2.position)

    # Sort edges by distance.
    shortest_edges = sorted(distance_dictionary, key=distance_dictionary.get)

    # Determine edges to add.
    new_edges = shortest_edges[:n_cross_edges]

    # Add edges to graph.
    for e in new_edges:
        new_measurement = create_measurement(combined_graph, e[0], e[1], translation_variance=translation_variance,
                                             rotation_variance=rotation_variance)
        combined_edges.append(new_measurement)

    # Write resulting graph to file and overwrite edges in combined graph object.
    write_g2o(combined_vertices, combined_edges, combined_path)
    combined_graph.edges = combined_edges

    return combined_graph


def create_multi_robot_dataset(n_cross_edges, box_widths=(10, 10), box_heights=(10, 10), box_distance=2, graph_1_path=None, graph_2_path=None, combined_path=None, edge_probability=0.05, translation_variance=0.2, rotation_variance=0.2, n_poses=20):

    # Set path strings if not included.
    if graph_1_path is None:
        graph_1_path = "datasets/robot_1_dataset.g2o"
    if graph_2_path is None:
        graph_2_path = "datasets/robot_2_dataset.g2o"
    if combined_path is None:
        combined_path = "datasets/combined_dataset.g2o"

    # Create first dataset.
    v_1, e_1 = create_geometric_dataset(translation_variance=translation_variance,
                                        rotation_variance=rotation_variance,
                                        n_poses=n_poses,
                                        file_name=graph_1_path,
                                        box=[0, box_widths[0], 0, box_heights[0]])

    # Create second dataset.
    v_2, e_2 = create_geometric_dataset(translation_variance=translation_variance,
                                        rotation_variance=rotation_variance,
                                        n_poses=n_poses,
                                        file_name=graph_2_path,
                                        box=[box_widths[0] + box_distance, box_widths[1] + box_widths[0] + box_distance, 0, box_heights[1]])

    # Define each graph from created datasets.
    graph_1 = Graph(vertices=v_1, edges=e_1)
    graph_2 = Graph(vertices=v_2, edges=e_2)

    # Create combined dataset.
    create_combined_dataset(graph_1, graph_2, n_cross_edges, combined_path=combined_path, translation_variance=translation_variance, rotation_variance=rotation_variance)

    # Weird bug with parsing graphs vs graph creation. Quick workaround is to just read them from file again.
    graph_1 = Graph(*parse_g2o(graph_1_path))
    graph_2 = Graph(*parse_g2o(graph_2_path))
    combined_graph = Graph(*parse_g2o(combined_path))

    return graph_1, graph_2, combined_graph


def multiple_robot_changes(graph_1, graph_2, combined_graph):

    # Find edges in combined graph that are not individual graphs.
    edge_interface = []
    for e in combined_graph.edges:
        if e.in_vertex < len(graph_1.vertices) <= e.out_vertex:
            edge_interface.append(e)
        elif e.in_vertex >= len(graph_1.vertices) > e.out_vertex:
            edge_interface.append(e)

    # Find distance of all poses from the critical edge.
    distances = graphical_distances(graph=combined_graph, edges=edge_interface)

    # Solve for all three graphs.
    graph_1_solution, graph_1_dual = pgo(graph_1)
    graph_2_solution, graph_2_dual = pgo(graph_2)
    combined_solution, combined_dual = pgo(combined_graph)

    # Print costs of each solution.
    print("Cost of graph 1 solution: " + str(cost_function(graph_1_solution)))
    print("Cost of graph 2 solution: " + str(cost_function(graph_2_solution)))
    print("Cost of combined solution: " + str(cost_function(combined_solution)))

    # Store magnitude of changes in dictionary with vertex id as key.
    graph_1_position_changes = {}
    graph_2_position_changes = {}
    combined_position_changes = {}
    rotation_changes = {}

    # Compute angle between first and second vertex in first graph to align.
    graph_1_angle = np.angle(graph_1_solution.vertices[1].position[0] + 1j*graph_1_solution.vertices[1].position[1])
    graph_1_combined_angle = np.angle(combined_solution.vertices[1].position[0] + 1j*combined_solution.vertices[1].position[1])
    graph_1_solution.rotate(graph_1_combined_angle - graph_1_angle)

    graph_2_angle = np.angle(graph_2_solution.vertices[1].position[0] + 1j * graph_2_solution.vertices[1].position[1])
    graph_2_anchor = combined_solution.vertices[len(graph_1_solution.vertices)].position[0] + 1j * combined_solution.vertices[len(graph_1_solution.vertices)].position[1]
    graph_2_compare = combined_solution.vertices[len(graph_1_solution.vertices) + 1].position[0] + 1j * combined_solution.vertices[len(graph_1_solution.vertices) + 1].position[1]
    graph_2_combined_angle = np.angle(graph_2_compare - graph_2_anchor)
    graph_2_solution.rotate(graph_2_combined_angle - graph_2_angle)

    # Find the magnitude of change for each pose in first graph.
    for i, v in enumerate(graph_1_solution.vertices):

        position_change = np.linalg.norm(v.position - combined_solution.vertices[i].position)
        rotation_change = np.mod(np.abs(v.rotation - combined_solution.vertices[i].rotation), np.pi)

        graph_1_position_changes[v.id] = position_change
        combined_position_changes[v.id] = position_change
        rotation_changes[v.id] = rotation_change

    # Find the magnitude of change for each pose in first graph.
    for i, v in enumerate(graph_2_solution.vertices):

        graph_2_anchor = combined_solution.vertices[len(graph_2_solution.vertices)].position

        position_change = np.linalg.norm(v.position - combined_solution.vertices[i + len(graph_1_solution.vertices)].position + graph_2_anchor)
        rotation_change = np.mod(np.abs(v.rotation - combined_solution.vertices[i + len(graph_1_solution.vertices)].rotation), np.pi)

        graph_2_position_changes[v.id] = position_change
        combined_position_changes[v.id + len(graph_2_solution.vertices)] = position_change
        rotation_changes[v.id + len(graph_2_solution.vertices)] = rotation_change

    # Create x and y data points for distance vs. magnitude change graph.
    x_data = []
    position_data = []
    rotation_data = []

    for key in combined_position_changes:
        x_data.append(distances[key])
        position_data.append(combined_position_changes[key])
        rotation_data.append(rotation_changes[key])

    plt.figure()
    plot_pose_graph(graph=graph_1_solution, color="red")

    plt.figure()
    plot_pose_graph(graph=graph_2_solution, offset=len(graph_1_solution.vertices))

    plt.figure()
    draw_combined_pose_graph(graph_1_solution, graph_2_solution, combined_solution)

    # plt.figure()
    # plot_pose_graph(graph=combined_solution, colors=position_changes)

    plt.figure()
    plot_pose_graph(graph=graph_1_solution, colors=graph_1_position_changes)

    plt.figure()
    plot_pose_graph(graph=graph_2_solution, colors=graph_2_position_changes)

    plt.figure()
    plot_pose_graph(graph=combined_solution, colors=combined_position_changes)

    plt.figure()
    plt.scatter(x_data, position_data)
    plt.xlabel("Distance from New Edges")
    plt.ylabel("Magnitude of Position Change")
    plt.title("Pose Position Change vs. Distance from New Edges")

    draw_plots()


def draw_combined_pose_graph(graph_1, graph_2, combined_graph):

    plot_pose_graph(combined_graph.subgraph([v.id for v in graph_1.vertices]), color='red')
    plot_pose_graph(combined_graph.subgraph([v.id + len(graph_1.vertices) for v in graph_2.vertices]))

    node_positions = {v.id: tuple(v.position) for v in combined_graph.vertices}

    cross_edges = []
    for e in combined_graph.edges:
        if e.in_vertex < len(graph_1.vertices) <= e.out_vertex:
            cross_edges.append((e.in_vertex, e.out_vertex))
        elif e.out_vertex < len(graph_1.vertices) <= e.in_vertex:
            cross_edges.append((e.in_vertex, e.out_vertex))

    g = nx.DiGraph()
    g.add_edges_from(cross_edges)
    g.add_nodes_from([v.id for v in combined_graph.vertices])

    nx.draw_networkx_edges(g, pos=node_positions, arrows=True, arrowsize=18)


def random_data_run(n_poses=60, n_cross_edges=15, box_distance=2, box_widths=(10, 10), box_heights=(10, 10)):

    # Dataset paths.
    dataset_1_path = "datasets/robot_1_dataset.g2o"
    dataset_2_path = "datasets/robot_2_dataset.g2o"
    combined_path = "datasets/combined_dataset.g2o"

    # "Good" dataset paths.
    # dataset_1_path = "datasets/good_robot_1_dataset.g2o"
    # dataset_2_path = "datasets/good_robot_2_dataset.g2o"
    # combined_path = "datasets/good_combined_dataset.g2o"

    # Individual graphs for debugging.
    # graph_1 = Graph(*parse_g2o(dataset_1_path))
    # graph_2 = Graph(*parse_g2o(dataset_2_path))
    # combined_graph = Graph(*parse_g2o(combined_path))

    # Create multiple datasets.
    graph_1, graph_2, combined_graph = create_multi_robot_dataset(n_cross_edges=n_cross_edges, n_poses=n_poses, box_distance=2,
                                                                  box_widths=(10, 10), box_heights=(10, 10),
                                                                  graph_1_path=dataset_1_path,
                                                                  graph_2_path=dataset_2_path,
                                                                  combined_path=combined_path)
    # Perform experiments on datasets.
    multiple_robot_changes(graph_1=graph_1, graph_2=graph_2, combined_graph=combined_graph)


def real_data_run():

    # Dataset paths.
    dataset_1_path = "datasets/input_MITb_half.g2o"
    dataset_2_path = "datasets/input_MITb_half.g2o"
    combined_path = "datasets/combined_real_dataset.g2o"

    # Read individual, real datasets.
    graph_1 = Graph(*parse_g2o(dataset_1_path))
    graph_2 = Graph(*parse_g2o(dataset_2_path))

    # Find right-most x-coordinate in graph_1 and left-most in graph_2.
    graph_1_right_coordinate = max([v.position[0] for v in graph_1.vertices])
    graph_2_left_coordinate = min([v.position[0] for v in graph_2.vertices])
    shift_vector = np.asarray([[10 + graph_1_right_coordinate - graph_2_left_coordinate], [0]])

    # Offset graph two so that the two graphs are disjoint.
    graph_2 = shift_graph(graph=graph_2, shift_vector=shift_vector)

    # Combine the two graphs.
    create_combined_dataset(graph_1, graph_2, translation_variance=0.01, rotation_variance=0.01, n_cross_edges=50, combined_path=combined_path)

    # Reread graph's - creating combined dataset mutates at least one of the graphs. Should be fixed at some point.
    graph_1 = Graph(*parse_g2o(dataset_1_path))
    graph_2 = Graph(*parse_g2o(dataset_2_path))
    combined_graph = Graph(*parse_g2o(combined_path))

    # Perform experiments on datasets.
    multiple_robot_changes(graph_1=graph_1, graph_2=graph_2, combined_graph=combined_graph)


if __name__ == "__main__":
    #random_data_run(n_poses=20, n_cross_edges=10, box_distance=2, box_widths=(10, 10), box_heights=(10, 10))
    real_data_run()

    # Path for generated file.
    FILE_PATH = "datasets/good_geometric_example.g2o"
