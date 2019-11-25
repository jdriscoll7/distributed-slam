from utility.parsing import parse_g2o
from matplotlib import pyplot as plt
import numpy as np


def _set_axes(x, y, xlim=None, ylim=None):

    if xlim is None:
        xlim = [1.1 ** (-1 * np.sign(np.min(x))) * np.min(x), 1.1 ** (np.sign(np.max(x))) * np.max(x)]

    if ylim is None:
        ylim = [1.1 ** (-1 * np.sign(np.min(y))) * np.min(y), 1.1 ** (np.sign(np.max(y))) * np.max(y)]

    plt.ylim(bottom=ylim[0], top=ylim[1])
    plt.xlim(left=xlim[0], right=xlim[1])


def plot_complex_list(data, xlim=None, ylim=None):

    # Extract coordinates for each vertex.
    x = [np.real(d) for d in data]
    y = [np.imag(d) for d in data]

    # Plot these pairs of coordinates.
    plt.figure()
    plt.plot(x, y, 'bo-')

    # Set plotting axes.
    _set_axes(x, y, xlim, ylim)


def plot_vertices(vertices, xlim=None, ylim=None):

    # Extract x coordinates of each vertex.
    x = [v.state[0] for v in vertices]
    y = [v.state[1] for v in vertices]

    # Plot these pairs of coordinates.
    plt.figure()
    plt.plot(x, y, 'bo-')

    # Set plotting axes.
    _set_axes(x, y, xlim, ylim)


def plot_pose_graph(vertices, edges, xlim=None, ylim=None):

    # Extract x coordinates of each vertex.
    x = [v.state[0] for v in vertices]
    y = [v.state[1] for v in vertices]

    # Plot these pairs of coordinates.
    plt.figure()

    # Need to repeatedly search vertex ids of "vertices" list.
    vertex_ids = [v.id for v in vertices]

    # Only draw edges corresponding to measurements.
    for edge in edges:

        # Find index of in and out vertices in "vertices" input list.
        in_vertex = vertex_ids.index(edge.in_vertex)
        out_vertex = vertex_ids.index(edge.out_vertex)

        # Plot this edge - change color if it is in spanning tree.
        if edge.out_vertex == edge.in_vertex - 1:
            plt.plot([x[out_vertex], x[in_vertex]], [y[out_vertex], y[in_vertex]], 'ro-')
        else:
            plt.plot([x[out_vertex], x[in_vertex]], [y[out_vertex], y[in_vertex]], 'ko-')

    # Set plotting axes.
    _set_axes(x, y, xlim, ylim)


def draw_plots():

    plt.show()


if __name__ == "__main__":
    vertices, edges = parse_g2o("../datasets/input_INTEL_g2o.g2o")