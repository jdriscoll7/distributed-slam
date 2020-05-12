from utility.parsing import parse_g2o
from matplotlib import pyplot as plt
import numpy as np


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


def plot_pose_graph(vertices=None, edges=None, xlim=None, ylim=None, graph=None, new_figure=True, ax=None):

    # Optional graph argument to specify vertices and edges.
    if graph is not None:
        vertices = graph.vertices
        edges = graph.edges

    # Extract x coordinates of each vertex.
    x = [v.position[0] for v in vertices]
    y = [v.position[1] for v in vertices]

    # Plot these pairs of coordinates.
    if new_figure:
        plt.figure()

    # Need to repeatedly search vertex ids of "vertices" list.
    vertex_ids = [v.id for v in vertices]

    # Only draw edges corresponding to measurements.
    for edge in edges:

        # Find index of in and out vertices in "vertices" input list.
        in_vertex = vertex_ids.index(edge.in_vertex)
        out_vertex = vertex_ids.index(edge.out_vertex)

        # If plot needs to be done on a certain axis, then do ax.plot rather than plt.plot.
        if ax is None:
            ax = plt

        # Plot this edge - change color if it is in spanning tree.
        if edge.out_vertex == edge.in_vertex - 1:
            # line, = ax.plot([x[out_vertex], x[in_vertex]], [y[out_vertex], y[in_vertex]], 'ro-', zorder=100)
            line, = ax.plot([x[out_vertex], x[in_vertex]], [y[out_vertex], y[in_vertex]], 'ko-', zorder=100, markersize=16)
        else:
            line, = ax.plot([x[out_vertex], x[in_vertex]], [y[out_vertex], y[in_vertex]], 'ko-', markersize=16)

    # Set plotting axes.
    if new_figure:
        _set_axes(x, y, xlim, ylim)

    return line


def draw_plots():

    plt.show()


if __name__ == "__main__":
    vertices, edges = parse_g2o("../datasets/input_INTEL_g2o.g2o")