import networkx as nx

from examples.paper_experiments.phase_sync.helper_functions import *


def difference_graph_with_colors(graph_1, graph_2, colors):

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
    plot_pose_graph(graph=graph_2.subgraph([v.id for v in graph_1.vertices]), colors=colors)

    # Find newly added vertices.
    small_graph_ids = [v.id for v in graph_1.vertices]
    new_vertex_ids = [v.id for v in graph_2.vertices if v.id not in small_graph_ids]
    new_vertex_pos = {v.id: tuple(v.position) for v in graph_2.vertices if v.id not in small_graph_ids}

    # Find edges in larger graph that are not in smaller graph.
    smaller_graph_edges = [(e.in_vertex, e.out_vertex) for e in graph_1.edges]
    added_edges = [(e.in_vertex, e.out_vertex) for e in graph_2.edges if
                   (e.in_vertex, e.out_vertex) not in smaller_graph_edges]

    vertex_neighborhood = [v.id for v in graph_2.subgraph(new_vertex_ids, neighborhood=True).vertices]

    # Initialize networkx graph.
    g = nx.DiGraph()
    g.add_edges_from(added_edges)
    g.add_nodes_from(vertex_neighborhood)

    node_positions = {v.id: tuple(v.position) for v in
                      graph_2.subgraph(new_vertex_ids, neighborhood=True).vertices}

    nx.draw_networkx_edges(g, pos=node_positions, arrows=True, arrowsize=18)
    nx.draw_networkx(g.subgraph(new_vertex_ids),
                     node_color='green',
                     pos=new_vertex_pos,
                     labels=dict((v_id, "$" + str(v_id + 1) + "$") for v_id in new_vertex_ids),
                     with_labels=True,
                     arrows=True,
                     arrowsize=18,
                     font_size=10,
                     node_size=400)

    plt.gca().set_xlim(left=left, right=right)
    plt.gca().set_ylim(bottom=bottom, top=top)


def graph_changes(graph_1, graph_2, reference_angle, focus_poses=None, final_graph=None):

    # Find edges in graph 2 that are not in graph 1.
    graph_1_edge_tuples = [(e.in_vertex, e.out_vertex) for e in graph_1.edges]
    added_edges = [e for e in graph_2.edges if (e.in_vertex, e.out_vertex) not in graph_1_edge_tuples]

    # Find distance of all poses from the critical edges.
    distances = graphical_distances(graph=graph_2, edges=added_edges)

    # Print costs of each solution.
    print("Cost of graph 1 solution: " + str(cost_function(graph_1)))
    print("Cost of graph 2 solution: " + str(cost_function(graph_2)))

    # Store magnitude of changes in dictionary with vertex id as key.
    position_changes = {}

    # Compute angle between first and second vertex in first graph to align.
    graph_1_angle = np.angle(graph_1.vertices[1].position[0] + 1j * graph_1.vertices[1].position[1])
    graph_2_angle = np.angle(graph_2.vertices[1].position[0] + 1j * graph_2.vertices[1].position[1])
    # graph_1.rotate(reference_angle - graph_1_angle)
    # graph_2.rotate(reference_angle - graph_2_angle)

    # Find the magnitude of change for each pose in first graph.
    position_change = 0
    solution_distance = 0
    for i, v in enumerate(graph_1.vertices):

        if v.id in focus_poses:

            position_change += np.linalg.norm(v.position - graph_2.get_vertex(v.id).position)
            solution_distance += np.linalg.norm(v.position - final_graph.get_vertex(v.id).position)

            print("Position change for vertex %d (distance %d): %f ([%f, %f] -> [%f, %f])" % (v.id, distances[v.id], position_change, v.position[0], v.position[1], graph_2.vertices[i].position[0], graph_2.vertices[i].position[1]))

        else:

            position_change += 0
            solution_distance += 0

    for i, v in enumerate(graph_1.vertices):
        if v.id in focus_poses:
            position_changes[v.id] = position_change
        else:
            position_changes[v.id] = 0

    # Create x and y data points for distance vs. magnitude change graph.
    x_data = []
    position_data = []

    distance_change_dictionary = {}

    for key in position_changes:
        x_data.append(distances[key])
        position_data.append(position_changes[key])
        distance_change_dictionary.setdefault(distances[key], []).append(position_changes[key])

    plt.figure()
    difference_graph_with_colors(graph_1, graph_2, colors=position_changes)
    # draw_plots()

    return graph_1, solution_distance


def real_data_run():

    graph = Graph(*parse_g2o("datasets/input_INTEL_g2o.g2o"))
    # graph = Graph(*parse_g2o("datasets/CSAIL_P.g2o"))

    graph_list = form_graph_sequence(graph, 0, step=20, order="id")

    start_point = 15
    n_graphs = 10

    solution_history = []
    changes_history = []

    reference_angle = np.angle(
        graph_list[start_point].vertices[1].position[0] + 1j * graph_list[start_point].vertices[1].position[1])
    solved_graphs = []

    # First solve all graphs.
    for i, g in enumerate(graph_list[start_point:(1 + start_point + n_graphs)]):
        solution, unused = pgo(g)
        solved_graphs.append(solution)

    last_graph = solved_graphs[0]

    for i, g in enumerate(graph_list[(1 + start_point):(1 + start_point + n_graphs)]):
        solution, change = graph_changes(graph_1=last_graph, graph_2=solved_graphs[i + 1],
                                         reference_angle=reference_angle,
                                         focus_poses=list(range(10, 20)),
                                         final_graph=solved_graphs[-2])
        solution_history.append(solution)
        changes_history.append(change)
        last_graph = solved_graphs[i]

    plt.figure()
    plt.plot(changes_history)
    plt.title("Distance from Block to Solution vs. Graph Time")
    plt.xlabel("Time (Number of Pose-Blocks added)")
    plt.ylabel("Change in Solution of Selected Block Pose")

    draw_plots()


def synthetic_data_run():
    # Dataset paths.
    dataset_path = "datasets/random_geometric_dataset.g2o"
    create_geometric_dataset(translation_variance=0.1,
                             rotation_variance=0.1,
                             n_poses=30,
                             file_name=dataset_path,
                             box=[0, 10, 0, 10])

    graph = Graph(*parse_g2o(dataset_path))
    graph_list = form_graph_sequence(graph, 0, step=4, order="distance")

    start_point = 2
    n_graphs = 3

    solution_history = []
    changes_history = []

    reference_angle = np.angle(graph_list[start_point].vertices[1].position[0] + 1j * graph_list[start_point].vertices[1].position[1])
    solved_graphs = []

    # First solve all graphs.
    for i, g in enumerate(graph_list[start_point:(1 + start_point + n_graphs)]):
        solution, unused = pgo(g)
        solved_graphs.append(solution)

    last_graph = solved_graphs[0]

    for i, g in enumerate(graph_list[(1 + start_point):(1 + start_point + n_graphs)]):
        solution, change = graph_changes(graph_1=last_graph, graph_2=solved_graphs[i + 1],
                                         reference_angle=reference_angle, focus_poses=list(range(10, 20)))
        solution_history.append(solution)
        changes_history.append(change)
        last_graph = solved_graphs[i]

    plt.figure()
    plt.scatter(changes_history)
    plt.title("Change in Selected Poses vs. Graph Time")
    plt.xlabel("Time (Number of Pose-Blocks added)")
    plt.ylabel("Change in Solution of Selected Block Pose")

    draw_plots()


if __name__ == "__main__":

    real_data_run()
    # synthetic_data_run()
