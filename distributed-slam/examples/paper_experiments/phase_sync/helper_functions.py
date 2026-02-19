import copy
import random

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


def form_graph_sequence(graph, pose_id, step=1, order="distance"):

    # Form list of vertices in order of distance from fixed pose.
    distance_dictionary = {}

    for v in graph.vertices:
        distance_dictionary[v.id] = np.linalg.norm(graph.get_vertex(pose_id).position - v.position)

    sorted_vertex_list = sorted(distance_dictionary, key=distance_dictionary.get)

    # Rearrange id's in graph so that id's are ascending w.r.t distance from input pose id.
    if order == "distance":
        graph.map_vertex_ids(sorted_vertex_list)

    # List of graphs to run optimization on.
    graph_list = []

    for i in range(1, len(sorted_vertex_list) // step):
        subgraph = graph.subgraph(i=[k for k in range(i*step)])
        graph_list.append(subgraph)

    return graph_list[1:]


def graphical_distances(graph, edges):
    """
    Given a graph and a particular edge, yields the distance of every vertex from the given edge or edges.

    :param graph: Input graph.
    :param edge:  Edge used to compute distances from.
    :return:      Dictionary consisting of vertex ids as keys and distances as values.
    """

    # To support multiple edges, need a list of distance dictionaries to take min over.
    distance_dictionary_list = []

    for i, edge in enumerate(edges):

        # Setup distance dictionary.
        distance_dictionary_list.append({edge.in_vertex: 0,
                                         edge.out_vertex: 0})

        # Keep track of distance being searched.
        distance = 1
        neighborhood = [edge.in_vertex, edge.out_vertex]

        # Iterate through neighborhoods until no vertices remain.
        while len(distance_dictionary_list[i]) < len(graph.vertices):
            neighborhood = [v.id for v in graph.neighborhood(neighborhood)]

            # Write distances for this neighborhood if they haven't been seen before.
            for v in neighborhood:
                if v not in distance_dictionary_list[i]:
                    distance_dictionary_list[i][v] = distance

            distance += 1

            if distance > len(graph.vertices):
                print("Graph is disconnected.")

    # Distance of poses to set of edges is the min over distance dictionary of each edge.
    distance_dictionary = {}
    for v in graph.get_vertices():
        distances = {}
        for d in distance_dictionary_list:
            if v.id in d.keys():
                distances.setdefault(v.id, []).append(d[v.id])
        distance_dictionary[v.id] = min(distances[v.id])

    return distance_dictionary


def edge_not_in_path(edge, n_vertices):
    return np.abs(edge.in_vertex - edge.out_vertex) > 1 and np.abs(edge.in_vertex - edge.out_vertex) != n_vertices - 1


def find_off_path_edge(graph):

    # for e in graph.edges:
    #     if edge_not_in_path(e, len(graph.vertices)):
    #         return e
    edges = graph.get_edges()
    random.shuffle(edges)

    return edges[0]
