from utility.parsing import parse_g2o
from matplotlib import pyplot as plt
import numpy as np
import networkx as nx


def _set_axes(x, y, xlim=None, ylim=None):

    if xlim is None:
        xlim = [-1 + 1.1 ** (-1 * np.sign(np.min(x))) * np.min(x), 1 + 1.1 ** (np.sign(np.max(x))) * np.max(x)]

    if ylim is None:
        ylim = [-1 + 1.1 ** (-1 * np.sign(np.min(y))) * np.min(y), 1 + 1.1 ** (np.sign(np.max(y))) * np.max(y)]

    # Make sure axes are not being made smaller than before.
    xmin, xmax, ymin, ymax = plt.axis()
    ylim[0] = ymin if ylim[0] > ymin else ylim[0]
    ylim[1] = ymax if ylim[1] < ymax else ylim[1]
    xlim[0] = xmin if xlim[0] > xmin else xlim[0]
    xlim[1] = xmax if xlim[1] < xmax else xlim[1]

    plt.ylim(bottom=ylim[0], top=ylim[1])
    plt.xlim(left=xlim[0], right=xlim[1])


def plot_complex_list(data, xlim=None, ylim=None):

    # Extract coordinates for each vertex.
    x = [np.real(d) for d in data]
    y = [np.imag(d) for d in data]

    # Plot these pairs of coordinates.
    plt.figure()
    ax = plt.axes()
    ax.set_xticks([])
    ax.set_yticks([])
    plt.plot(x, y, 'bo-', markersize=1)

    # Set plotting axes.
    _set_axes(x, y, xlim, ylim)


def plot_graphs(graphs, new_figure=False):

    for g in graphs:
        plot_graph(graph=g, new_figure=new_figure)


def plot_graph(graph, new_figure=False):

    plot_vertices(graph.vertices, new_figure=new_figure, edges=graph.edges, labels=True)


def plot_vertices(vertices, xlim=None, ylim=None, new_figure=True, color=None, edges=[], labels=False):

    # Extract x coordinates of each vertex.
    x = [v.position[0] for v in vertices]
    y = [v.position[1] for v in vertices]

    # Make new figure and get current axes.
    ax = None
    if new_figure is True:
        plt.figure()

    ax = plt.gca()

    # Choose color.
    if color is None:
        color = np.random.rand(3, )

    # Plot these pairs of coordinates.
    ax.scatter(x, y, marker='o', s=4, c=color)

    # Annotate plot with vertex numbers.
    if labels:
        for v in vertices:
            ax.annotate('%d' % v.id, xy=tuple(v.position), textcoords='data')

    # Plot rotations.
    for i, xy in enumerate(zip(x, y)):
        rotation = vertices[i].rotation
        ax.arrow(xy[0], xy[1], np.cos(rotation), np.sin(rotation), head_width=.05)

    # Plot relative measurements if edges are included.
    for e in edges:

        # Current x and y positions and their offsets from relative measurements.
        index = np.argwhere([e.out_vertex == v.id for v in vertices])[0][0]
        base_x = x[index]
        base_y = y[index]

        offset_x = e.relative_pose[0]
        offset_y = e.relative_pose[1]

        ax.plot([base_x, base_x + offset_x],
                [base_y, base_y + offset_y],
                '--')

        ax.annotate('(%s, %s)' % (e.out_vertex, e.in_vertex),
                    xy=(base_x + offset_x / 2, base_y + offset_y / 2),
                    textcoords='data')

    # Set plotting axes.
    _set_axes(x, y, xlim, ylim)


def plot_pose_graph(graph, colors=None, color=None, offset=0):

    # Collect edges as tuples between vertex ids.
    edges = [(e.out_vertex + offset, e.in_vertex + offset) for e in graph.edges]

    # Initialize networkx graph.
    g = nx.DiGraph()
    g.add_edges_from(edges)

    # Collect positions of vertices.
    positions = {v.id + offset: tuple(v.position) for v in graph.vertices}

    # Generate labels for vertices.
    labels = {v.id + offset: "$" + str(v.id + 1 + offset) + "$" for v in graph.vertices}

    # Generate colors if needed.
    if colors is not None:
        color_list = [colors[v - offset] for v in g.nodes()]

    # Draw graph.
    if colors is not None:
        cmap = plt.get_cmap('plasma')

        nx.draw_networkx(g, cmap=cmap, node_color=color_list, pos=positions, labels=labels, with_labels=True, arrows=True, arrowsize=18, font_size=10)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(color_list), vmax=max(color_list)))
        sm._A = []
        cb = plt.colorbar(sm, label="Magnitude in Position Change")
        cb.ax.set_yticklabels(["{:.3f}".format(i) for i in cb.get_ticks()])
    else:
        if color is None:
            nx.draw_networkx(g, pos=positions, labels=labels, with_labels=True, arrows=True, arrowsize=18, font_size=10)
        else:
            nx.draw_networkx(g, pos=positions, node_color=color, labels=labels, with_labels=True, arrows=True, arrowsize=18, font_size=10)


def plot_line_graph(graph):

    # Collect edges as tuples between vertex ids.
    edges = [(e.out_vertex, e.in_vertex) for e in graph.edges]

    # Initialize networkx graph.
    g = nx.DiGraph()
    g.add_edges_from(edges)

    # Form line graph.
    g = nx.line_graph(g.to_undirected())

    # Collect positions of vertices.
    positions = {}
    for v_1 in graph.vertices:
        for v_2 in graph.vertices:
            positions[(v_1.id, v_2.id)] = tuple(0.5 * (v_1.position + v_2.position))

    labels = {}
    for e in edges:
        labels[e] = "$" + str(e[1] + 1) + "-" + str(e[0] + 1) + "$"

    # Draw graph.
    nx.draw_networkx(g, pos=positions, labels=labels, with_labels=True, arrows=True, arrowsize=18, font_size=5)


def draw_plots():

    plt.show()


if __name__ == "__main__":
    vertices, edges = parse_g2o("../datasets/input_INTEL_g2o.g2o")