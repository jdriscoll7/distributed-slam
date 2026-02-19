import networkx as nx
import os

from examples.paper_experiments.phase_sync.helper_functions import *


def solution_comparison_alignment(small_graph, large_graph, small_to_large_map):

    large_subgraph = large_graph.subgraph([small_to_large_map[v.id] for v in small_graph.vertices], reduce=True)[0]

    small_state = small_graph.get_complex_state()
    large_state = large_subgraph.get_complex_state()

    # cost_f = lambda rot: np.max(np.abs(np.exp(-1j*rot) * small_state - large_state))
    cost_f = lambda rot: np.linalg.norm(np.exp(-1j * rot) * small_state - large_state)
    cost_f_vec = np.vectorize(cost_f)

    rotations = np.linspace(-np.pi/2, np.pi/2, 1000)
    cost_evals = cost_f_vec(rotations)
    min_index = np.argmin(cost_evals)
    best_rotation = rotations[min_index]

    # Grid search.
    # rotation_grid = [-np.pi/2, np.pi/2]
    # cost_grid = [cost_f(rotation_grid[0]), cost_f(rotation_grid[1])]

    # Evaluate mid point of grid.
    # cost = cost_f(rotation_grid[1] / 2 - rotation_grid[0] / 2)
    #
    # for i in range(200):
    #
    #     if cost < cost_grid[0]:
    #         cost_grid[0] = cost
    #         rotation_grid[0] = rotation_grid[0] / 2 - rotation_grid[1] / 2
    #     else:
    #         cost_grid[1] = cost
    #         rotation_grid[1] = rotation_grid[0] / 2 - rotation_grid[1] / 2
    #
    #     # Evaluate mid point of grid.
    #     cost = cost_f(rotation_grid[1] / 2 - rotation_grid[0] / 2)

    # small_graph.rotate(rotation_grid[1] / 2 - rotation_grid[0] / 2)
    small_graph.rotate(best_rotation)

    return small_graph, large_graph


def difference_graph_with_colors(graph_1, graph_2, colors, with_labels=False, small_to_large_map=None):

    # Find plotting bounds to nicely fit frames.
    left = 0
    right = 0
    bottom = 0
    top = 0

    for graph in [graph_1, graph_2]:
        # Extract maximal coordinates.
        x_min = np.min(np.real(graph.get_complex_state()))
        x_max = np.max(np.real(graph.get_complex_state()))
        y_min = np.min(np.imag(graph.get_complex_state()))
        y_max = np.max(np.imag(graph.get_complex_state()))

        # Keep track of minimal/maximal coordinates.
        left = left if left < x_min else x_min
        right = right if right > x_max else x_max
        bottom = bottom if bottom < y_min else y_min
        top = top if top > y_max else y_max

    left = left - 0.1 * np.abs(left) - 1
    right = right + 0.1 * np.abs(right) + 1
    top = top + 0.1 * np.abs(top) + 1
    bottom = bottom - 0.1 * np.abs(bottom) - 1

    # Draw pose graph with heat map.
    plot_pose_graph(graph=graph_1, colors=colors, with_edges=False)

    # Find newly added vertices.
    if small_to_large_map is None:
        small_graph_ids = [v.id for v in graph_1.vertices]
        new_vertex_ids = [v.id for v in graph_2.vertices if v.id not in small_graph_ids]
        new_vertex_pos = {v.id: tuple(v.position) for v in graph_2.vertices if v.id not in small_graph_ids}

        # Find edges in larger graph that are not in smaller graph.
        smaller_graph_edges = [(e.in_vertex, e.out_vertex) for e in graph_1.edges]
        added_edges = [(e.in_vertex, e.out_vertex) for e in graph_2.edges if
                       (e.in_vertex, e.out_vertex) not in smaller_graph_edges]
    else:
        small_graph_ids = [small_to_large_map[v.id] for v in graph_1.vertices]
        new_vertex_ids = [v.id for v in graph_2.vertices if v.id not in small_graph_ids]
        new_vertex_pos = {v.id: tuple(v.position) for v in graph_2.vertices if v.id not in small_graph_ids}

        # Find edges in larger graph that are not in smaller graph.
        smaller_graph_edges = [(small_to_large_map[e.in_vertex], small_to_large_map[e.out_vertex]) for e in graph_1.edges]
        added_edges = [(e.in_vertex, e.out_vertex) for e in graph_2.edges if
                       (e.in_vertex, e.out_vertex) not in smaller_graph_edges]

    vertex_neighborhood = [v.id for v in graph_2.subgraph(new_vertex_ids, neighborhood=True).vertices]

    # Initialize networkx graph.
    g = nx.DiGraph()
    g.add_edges_from(added_edges)
    # g.add_nodes_from(vertex_neighborhood)

    node_positions = {v.id: tuple(v.position) for v in
                      # graph_2.subgraph(new_vertex_ids, neighborhood=True).vertices}
                      graph_2.vertices}

    nx.draw_networkx_edges(g, pos=node_positions, arrows=False, arrowsize=18)
    nx.draw_networkx(g.subgraph(new_vertex_ids),
                     node_color='green',
                     pos=new_vertex_pos,
                     labels=dict((v_id, "$" + str(v_id + 1) + "$") for v_id in new_vertex_ids),
                     with_labels=with_labels,
                     arrows=True,
                     arrowsize=18,
                     font_size=10,
                     node_size=400)

    plt.gca().set_xlim(left=left, right=right)
    plt.gca().set_ylim(bottom=bottom, top=top)


def graph_changes(graph_1, graph_2, reference_angle, vertex_map=None):

    small_to_large_map = dict((v, k) for k, v in vertex_map.items())
    large_to_small_map = vertex_map

    # Find edges in graph 2 that are not in graph 1.

    graph_1_edge_tuples = [(small_to_large_map[e.in_vertex], small_to_large_map[e.out_vertex]) for e in graph_1.edges]
    added_edges = [e for e in graph_2.edges if (e.in_vertex, e.out_vertex) not in graph_1_edge_tuples]

    graph_1_angle = np.angle(graph_1.vertices[1].position[0] + 1j * graph_1.vertices[1].position[1])
    graph_2_angle = np.angle(graph_2.vertices[1].position[0] + 1j * graph_2.vertices[1].position[1])

    # graph_1_angle = graph_1.vertices[0].rotation
    # graph_2_angle = graph_2.vertices[0].rotation

    graph_1, graph_2 = solution_comparison_alignment(small_graph=graph_1, large_graph=graph_2, small_to_large_map=small_to_large_map)

    print(graph_1_angle)
    print(graph_2_angle)
    # graph_1.rotate(reference_angle - graph_1_angle)
    # graph_2.rotate(reference_angle - graph_2_angle)
    # graph_1.rotate(reference_angle - graph_1_angle)
    # graph_2.rotate(reference_angle - graph_2_angle)

    # Find distance of all poses from the critical edges.
    distances = graphical_distances(graph=graph_2, edges=added_edges)

    # Print costs of each solution.
    print("Cost of graph 1 solution: " + str(cost_function(graph_1)))
    print("Cost of graph 2 solution: " + str(cost_function(graph_2)))

    # Store magnitude of changes in dictionary with vertex id as key.
    position_changes = {}
    position_changes_small_map = {}
    rotation_changes = {}

    # Find the magnitude of change for each pose in first graph.
    for i, v in enumerate(graph_1.vertices):
        # print([x.id for x in graph_1.vertices])
        # print([x.id for x in graph_2.vertices])
        graph_2_vertex = graph_2.get_vertex(small_to_large_map[v.id])
        position_change = np.linalg.norm(v.position - graph_2_vertex.position)
        # rotation_change = np.mod(np.abs(v.rotation - graph_2_vertex.rotation), np.pi)
        rotation_change = np.linalg.norm(np.exp(-1j*v.rotation) - np.exp(-1j*graph_2_vertex.rotation))

        print("Position change for vertex %d (distance %d): %f ([%f, %f] -> [%f, %f])" % (v.id, distances[v.id], position_change, v.position[0], v.position[1], graph_2_vertex.position[0], graph_2_vertex.position[1]))

        position_changes[small_to_large_map[v.id]] = position_change
        position_changes_small_map[v.id] = position_change
        rotation_changes[small_to_large_map[v.id]] = rotation_change

    # Create x and y data points for distance vs. magnitude change graph.
    x_data = []
    position_data = []
    rotation_data = []

    distance_change_dictionary = {}

    for key in position_changes:
        x_data.append(distances[key])
        position_data.append(position_changes[key])
        rotation_data.append(rotation_changes[key])
        distance_change_dictionary.setdefault(distances[key], []).append(position_changes[key])

    # Data averages.
    change_maxes = []
    max_x_data = []
    for key in distance_change_dictionary:
        max_x_data.append(key)
        change_maxes.append(np.max(distance_change_dictionary[key]))
    #

    # plt.figure()
    # difference_graph_with_colors(graph_1, graph_2, colors=position_changes_small_map, small_to_large_map=small_to_large_map)

    for i in range(len(graph_2.vertices)):
        if i not in position_changes:
            position_changes[i] = -1


    #
    # plt.figure()
    # plot_pose_graph(graph=graph_1, colors=position_changes)

    # plt.figure()
    # plot_pose_graph(graph=graph_2)
    #
    # plt.figure()
    sorted_idx = np.argsort(max_x_data)
    print(sorted_idx)
    print(max_x_data)
    print(change_maxes)
    plt.plot(np.array(max_x_data)[sorted_idx], np.log(np.array(change_maxes)[sorted_idx] + 0.001))
    # plt.plot(np.array(max_x_data)[sorted_idx], np.array(change_maxes)[sorted_idx])
    # plt.scatter(np.array(max_x_data), np.array(change_maxes))
    plt.title("Largest Position Change in Graph vs. Distance")
    plt.xlabel("Solution Distance in Graph")
    plt.ylabel("Magnitude of Solution Change")

    return graph_1, position_changes


def real_data_run():

    print(os.listdir())
    dataset_path = "../../../datasets/input_MITb_g2o.g2o"
    # dataset_path = "datasets/random_geometric_dataset.g2o"

    graph = Graph(*parse_g2o(dataset_path))
    n_poses = len(graph.vertices)
    graph_list = []

    for i in range(1, n_poses):
        graph_list.append(Graph(*parse_g2o(dataset_path)).subgraph(i=[k for k in range(n_poses) if k != i], reduce=True))

    # graph_list = [graph_list[563]]
    graph_list = [graph_list[600]]

    solution_history = []
    changes_history = []

    reference_angle = np.angle(graph.vertices[1].position[0] + 1j * graph.vertices[1].position[1])
    solved_graphs = []

    main_solution, unused = pgo(graph)

    # First solve all graphs.
    for i, g in enumerate(graph_list):
        solution, unused = pgo(g[0])
        solved_graphs.append((solution, g[1]))

    for i in range(len(solved_graphs)):
        solution, change = graph_changes(graph_1=solved_graphs[i][0], graph_2=main_solution,
                                         reference_angle=reference_angle, vertex_map=solved_graphs[i][1])
        solution_history.append(solution)
        changes_history.append(change)

    draw_plots()


def synthetic_data_run(n_poses=100):
    # Dataset paths.
    dataset_path = "datasets/random_geometric_dataset.g2o"
    create_geometric_dataset(translation_variance=0.1,
                             rotation_variance=0.001,
                             n_poses=n_poses,
                             file_name=dataset_path,
                             box=[0, 20, 0, 20])

    graph = Graph(*parse_g2o(dataset_path))
    graph_list = []

    for i in range(1, n_poses):
        graph_list.append(Graph(*parse_g2o(dataset_path)).subgraph(i=[k for k in range(n_poses) if k != i], reduce=True))

    graph_list = graph_list[10:-10:18]
    # graph_list = [graph_list[-]]

    solution_history = []
    changes_history = []

    solved_graphs = []

    main_solution, unused = pgo(graph)
    reference_angle = np.angle(graph.vertices[1].position[0] + 1j * graph.vertices[1].position[1])
    # reference_angle = graph.vertices[0].rotation

    # First solve all graphs.
    for i, g in enumerate(graph_list):
        solution, unused = pgo(g[0])
        solved_graphs.append((solution, g[1]))

    for i in range(len(solved_graphs)):
        solution, change = graph_changes(graph_1=solved_graphs[i][0], graph_2=main_solution,
                                         reference_angle=reference_angle, vertex_map=solved_graphs[i][1])
        solution_history.append(solution)
        changes_history.append(change)

    draw_plots()


if __name__ == "__main__":

    # real_data_run()
    synthetic_data_run(n_poses=100)
