import copy

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from os import makedirs
from os.path import dirname


from solvers.sdp import pgo
from utility.data_generation import serial_graph_plan
from utility.data_generation import create_random_dataset
from utility.graph import Graph
from utility.visualization import plot_complex_list, draw_plots, plot_pose_graph
from utility.parsing import parse_g2o
import matplotlib.gridspec as gridspec

from solvers.admm.local_admm import cost_function


def _create_distance_matrix(data, solution):

    # Determine individual trajectories/histories.
    distances = np.array([[x[n] if len(x) > n else None for x in data] for n in range(len(data[-1]))])

    # Determine distances.
    for i in range(distances.shape[0]):
        distances[i, distances[i, :] != None] = np.abs(distances[i, distances[i, :] != None] - solution[i])

    return distances


def create_gif(update, data, data_length, name):

    plt.figure()
    fig, ax = plt.subplots(1, 1)
    fig.subplots_adjust(top=1.0,
                        bottom=0.0,
                        left=0.0,
                        right=1.0,
                        hspace=0.0,
                        wspace=0.0)

    for location in ["left", "right", "top", "bottom"]:
        ax.spines[location].set_visible(False)

    ani = animation.FuncAnimation(fig, update, data_length, fargs=[data, ax], interval=1000, blit=True)

    # Setup mp4 writing.
    writer = animation.writers['ffmpeg']
    writer = writer(fps=0.5, metadata=dict(artist='Me'), bitrate=1800)

    # Write the mp4.
    ani.save(name, writer=writer)


def create_trajectory_gif(data, solution, graph_history, rotation_history=None):

    # Make sure output directory exists - if not, create it.
    if dirname('trajectory_files/trajectory.mp4') is not '':
        makedirs(dirname('trajectory_files/trajectory.mp4'), exist_ok=True)

    solution = [[np.real(x) for x in solution], [np.imag(x) for x in solution]]

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

    def update(n, data, ax):

        for a in (ax if isinstance(ax, tuple) else [ax]):
            a.clear()
            a.set_xlim(left=left, right=right)
            a.set_ylim(bottom=bottom, top=top)
            #a.set_aspect('equal')
            a.set_xticks([])
            a.set_yticks([])

        # Extract coordinates for each vertex.
        x = [np.real(d) for d in data[n]]
        y = [np.imag(d) for d in data[n]]

        cost_function(graph_history[n])

        # line_1, = ax.plot(x, y, 'bo-', markersize=2, linewidth=1, zorder=2)
        # line_2, = ax.plot(solution[0], solution[1], 'ro-', linewidth=2, markersize=4, zorder=1)
        line = plot_pose_graph(vertices=graph_history[n].vertices, edges=graph_history[n].edges, new_figure=False, ax=ax)

        # Save individual frames.
        plt.savefig('trajectory_files/%d.png' % (n))

        return line,

    create_gif(update, data, len(data), 'trajectory_files/trajectory.mp4')


def create_distance_gif(data, solution):

    distances = _create_distance_matrix(data, solution)

    # Extract coordinates to plot solution.
    solution = [[np.real(x) for x in solution], [np.imag(x) for x in solution]]

    def update(n, distance_data, ax):
        ax.clear()
        ax.set_xlim(left=-12, right=12)
        ax.set_ylim(bottom=-12, top=12)

        for i in range(distance_data.shape[0]):
            if distance_data[i, n] is not None:
                ax.add_artist(plt.Circle((solution[0][i], solution[1][i]), distance_data[i, n] + 0.1, fill=False))

        line = ax.scatter(solution[0], solution[1], marker='o', color='r', s=8)

        return line,

    create_gif(update, distances, distances.shape[1], 'distance_long.gif')


def plot_vertex_trajectory(data, solution):

    # Make a new figure.
    plt.figure()
    plt.title('Vertex Trajectory')

    # Determine individual trajectories/histories.
    trajectories = [[x[n] for x in data if len(x) > n] for n in range(len(data[-1]))]

    solution = [[np.real(x) for x in solution], [np.imag(x) for x in solution]]

    # Plot the trajectories.
    for t in trajectories:

        # Extract coordinates for each vertex.
        x = [np.real(x) for x in t]
        y = [np.imag(x) for x in t]

        plt.plot(x, y, 'bo-', markersize=2, linewidth=1)
        plt.scatter(solution[0], solution[1], marker='o', color='r', s=2)
        plt.xlim(left=-12, right=12)
        plt.ylim(bottom=-12, top=12)


def plot_vertex_distances(data, solution):

    # Make a new figure.
    fig, ax = plt.subplots()
    ax.set_title('Vertex Distances')

    # Determine individual trajectories/histories.
    distances = np.array([[x[n] if len(x) > n else None for x in data] for n in range(len(data[-1]))])

    # Determine distances.
    for i in range(np.size(distances, 0)):
        distances[i, distances[i, :] != None] = np.abs(distances[i, distances[i, :] != None] - solution[i])

    # Extract coordinates to plot solution.
    solution = [[np.real(x) for x in solution], [np.imag(x) for x in solution]]

    # Plot the trajectories.
    for j in [10]:
        for i in range(distances.shape[1]):
            if distances[i, j] is not None:
                ax.add_artist(plt.Circle((solution[0][i], solution[1][i]), distances[i, j] + 0.1, fill=False))
                plt.xlim(left=-12, right=12)
                plt.ylim(bottom=-12, top=12)
        plt.scatter(solution[0], solution[1], marker='o', color='r', s=8)


if __name__ == "__main__":

    # Path for generated file.
    FILE_PATH = "/home/joe/repositories/distributed-slam/datasets/input_INTEL_g2o.g2o"

    # Number of vertices in generated data and starting point for test.
    # N_VERTICES = 10
    # STARTING_VERTEX = 2
    #
    # # Create random dataset.
    # vertex_list, edge_list = create_random_dataset(0.1, 0.1, 10, "generated_data/test.g2o")
    #
    # solutions = []
    #
    # for (vertices, edges) in serial_graph_plan(vertex_list, edge_list, [5, 6, 7, 8, 9, 10]):
    #     solutions.append(pgo(vertices, edges)[0])
    #
    # create_trajectory_gif(solutions, solutions[-1])


    # Parse data set.
    vertex_list, edge_list = parse_g2o(FILE_PATH)

    solutions = []

    plan_sizes = list(range(50, 200, 10))

    # Keep track of vertices and edges used.
    graph_history = []
    rotation_solutions = []

    for (vertices, edges) in serial_graph_plan(vertex_list, edge_list, list(plan_sizes)):

        # Add to list of solutions.
        positions, rotations = pgo(vertices, edges)[0:2]

        solutions.append(positions)
        rotation_solutions.append(rotations)

        # Add to list of vertices and edges used.
        graph_history.append(Graph(vertices, edges))
        graph_history[-1].update_states(np.vstack((positions, rotations)))

    create_trajectory_gif(solutions, solutions[-1], graph_history, rotation_solutions)

    # Extract coordinates for each vertex.
    x = [np.real(d) for d in solutions[-1]]
    y = [np.imag(d) for d in solutions[-1]]
    for i in range(len(solutions)):
        solutions[i] = solutions[-1][:len(solutions[i])]

    create_trajectory_gif(solutions, solutions[-1], graph_history, rotation_solutions)
    create_distance_gif(solutions, solutions[-1])

    plot_vertex_trajectory(solutions, solutions[-1])
    plot_pose_graph(vertex_list, edge_list, xlim=[-12, 12], ylim=[-12, 12])
    draw_plots()
