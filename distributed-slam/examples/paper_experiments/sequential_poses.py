import networkx as nx

from helper_functions import *
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter


def create_gif(update, data, data_length, name):

    fig, ax = plt.subplots(1, 1)
    # fig.subplots_adjust(top=1.0,
    #                     bottom=0.0,
    #                     left=0.0,
    #                     right=1.0,
    #                     hspace=0.0,
    #                     wspace=0.0)

    for location in ["left", "right", "top", "bottom"]:
        ax.spines[location].set_visible(False)

    ani = animation.FuncAnimation(fig, update, data_length, fargs=[data, ax], interval=1000, blit=False)

    # Setup mp4 writing.
    writer = animation.writers['ffmpeg']
    writer = writer(fps=0.5, metadata=dict(artist='Me'), bitrate=1800)

    # Write the mp4.
    ani.save(name, writer=writer)


def create_trajectory_gif(graph_history, changes_history, last_graph, rotation_history=None):

    # Make sure output directory exists - if not, create it.
    if dirname('trajectory_files/trajectory.mp4') != '':
        makedirs(dirname('trajectory_files/trajectory.mp4'), exist_ok=True)

    # Find plotting bounds to nicely fit frames.
    left = 0
    right = 0
    bottom = 0
    top = 0

    for graph in graph_history:
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

    def update(n, graph_change_history, ax):

        plt.clf()

        cost_function(graph_history[n])

        # line_1, = ax.plot(x, y, 'bo-', markersize=2, linewidth=1, zorder=2)
        # line_2, = ax.plot(solution[0], solution[1], 'ro-', linewidth=2, markersize=4, zorder=1)

        # Draw pose graph with heat map.
        plot_pose_graph(graph=graph_change_history[0][n], colors=graph_change_history[1][n])

        # Find newly added vertex.
        if n + 2 > len(graph_change_history[0]):
            larger_graph = last_graph
        else:
            larger_graph = graph_change_history[0][n + 1]

        small_graph_ids = [v.id for v in graph_history[n].vertices]
        new_vertex_ids = [v.id for v in larger_graph.vertices if v.id not in small_graph_ids]
        new_vertex_pos = {v.id: tuple(v.position) for v in larger_graph.vertices if v.id not in small_graph_ids}

        # Find edges in larger graph that are not in smaller graph.
        smaller_graph_edges = [(e.in_vertex, e.out_vertex) for e in graph_history[n].edges]
        added_edges = [(e.in_vertex, e.out_vertex) for e in larger_graph.edges if (e.in_vertex, e.out_vertex) not in smaller_graph_edges]

        vertex_neighborhood = [v.id for v in larger_graph.subgraph(new_vertex_ids, neighborhood=True).vertices]

        # Initialize networkx graph.
        g = nx.DiGraph()
        g.add_edges_from(added_edges)
        g.add_nodes_from(vertex_neighborhood)

        node_positions = {v.id: tuple(v.position) for v in larger_graph.subgraph(new_vertex_ids, neighborhood=True).vertices}

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

        # for a in (ax if isinstance(ax, tuple) else [ax]):
        #     a.clear()
        #     a.set_xlim(left=left, right=right)
        #     a.set_ylim(bottom=bottom, top=top)
        #     #a.set_aspect('equal')
        #     # a.set_xticks([])
        #     # a.set_yticks([])

        line = plt.gcf().get_children()
        plt.gca().set_xlim(left=left, right=right)
        plt.gca().set_ylim(bottom=bottom, top=top)

        # Save individual frames.
        plt.savefig('trajectory_files/%d.png' % (n))

        return line,

    create_gif(update, [graph_history, changes_history], len(graph_history), 'trajectory_files/trajectory.mp4')


def graph_changes(graph_1, graph_2, reference_angle):

    # Find edges in graph 2 that are not in graph 1.
    graph_1_edge_tuples = [(e.in_vertex, e.out_vertex) for e in graph_1.edges]
    added_edges = [e for e in graph_2.edges if (e.in_vertex, e.out_vertex) not in graph_1_edge_tuples]

    # Find distance of all poses from the critical edge.
    distances = graphical_distances(graph=graph_2, edges=added_edges)

    # Print costs of each solution.
    print("Cost of graph 1 solution: " + str(cost_function(graph_1)))
    print("Cost of graph 2 solution: " + str(cost_function(graph_2)))

    # Store magnitude of changes in dictionary with vertex id as key.
    position_changes = {}
    rotation_changes = {}

    # Compute angle between first and second vertex in first graph to align.
    graph_1_angle = np.angle(graph_1.vertices[1].position[0] + 1j * graph_1.vertices[1].position[1])
    graph_2_angle = np.angle(graph_2.vertices[1].position[0] + 1j * graph_2.vertices[1].position[1])
    # graph_1.rotate(reference_angle - graph_1_angle)
    # graph_2.rotate(reference_angle - graph_2_angle)

    # Find the magnitude of change for each pose in first graph.
    for i, v in enumerate(graph_1.vertices):

        position_change = np.linalg.norm(v.position - graph_2.vertices[i].position)
        rotation_change = np.mod(np.abs(v.rotation - graph_2.vertices[i].rotation), np.pi)
        print("Position change for vertex %d: %f ([%f, %f] -> [%f, %f])" % (v.id, position_change, v.position[0], v.position[1], graph_2.vertices[i].position[0], graph_2.vertices[i].position[1]))

        position_changes[v.id] = position_change
        rotation_changes[v.id] = rotation_change

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
    change_averages = []
    average_x_data = []
    for key in distance_change_dictionary:
        average_x_data.append(key)
        change_averages.append(np.mean(distance_change_dictionary[key]))

    # plt.figure()
    # plot_pose_graph(graph=graph_1)
    #
    # plt.figure()
    # plot_pose_graph(graph=graph_2)

    plt.figure()
    plot_pose_graph(graph=graph_1, colors=position_changes)

    plt.figure()
    plt.scatter(x_data, position_data)
    plt.title("Position Change in Graph vs. Distance")

    plt.figure()
    plt.scatter(average_x_data, change_averages)
    plt.title("Average Position Change in Graph vs. Distance")

    return graph_1, position_changes


if __name__ == "__main__":

    # Dataset paths.
    # dataset_path = "datasets/random_geometric_dataset.g2o"
    # create_geometric_dataset(translation_variance=0.1,
    #                          rotation_variance=0.1,
    #                          n_poses=30,
    #                          file_name=dataset_path,
    #                          box=[0, 10, 0, 10])

    # graph = Graph(*parse_g2o(dataset_path))
    # graph = Graph(*parse_g2o("datasets/input_INTEL_g2o.g2o"))
    graph = Graph(*parse_g2o("datasets/CSAIL_P.g2o"))

    graph_list = form_graph_sequence(graph, 0, step=20, order="id")

    start_point = 40
    n_graphs = 6
    last_graph = graph_list[start_point]

    solution_history = []
    changes_history = []

    reference_angle = np.angle(graph_list[start_point].vertices[1].position[0] + 1j * graph_list[start_point].vertices[1].position[1])
    solved_graphs = []

    # First solve all graphs.
    for i, g in enumerate(graph_list[(1 + start_point):(1 + start_point + n_graphs)]):
        solution, unused = pgo(g)
        solved_graphs.append(solution)

    for i, g in enumerate(graph_list[(1 + start_point):(1 + start_point + n_graphs)]):
        solution, change = graph_changes(graph_1=last_graph, graph_2=solved_graphs[i],
                                         reference_angle=reference_angle)
        solution_history.append(solution)
        changes_history.append(change)
        last_graph = solved_graphs[i]

    create_trajectory_gif(solution_history, changes_history, last_graph)

    draw_plots()