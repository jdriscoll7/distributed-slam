import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


from solvers.sdp import pgo
from perturbations import serial_graph_plan
from utility.data_generation import create_random_dataset
from utility.visualization import plot_complex_list, draw_plots, plot_pose_graph


def _create_distance_matrix(data, solution):

    # Determine individual trajectories/histories.
    distances = np.array([[x[n] if len(x) > n else None for x in data] for n in range(len(data[-1]))])

    # Determine distances.
    for i in range(distances.shape[0]):
        distances[i, distances[i, :] != None] = np.abs(distances[i, distances[i, :] != None] - solution[i])

    return distances


def create_gif(update, data, data_length, name):

    plt.figure()
    fig, ax = plt.subplots()
    ax.set_xlim(left=-12, right=12)
    ax.set_ylim(bottom=-12, top=12)

    ani = animation.FuncAnimation(fig, update, data_length, fargs=[data, ax], interval=500, blit=True)
    ani.save(name)


def create_trajectory_gif(data, solution):

    solution = [[np.real(x) for x in solution], [np.imag(x) for x in solution]]

    def update(n, data, ax):
        ax.clear()
        ax.set_xlim(left=-12, right=12)
        ax.set_ylim(bottom=-12, top=12)

        # Extract coordinates for each vertex.
        x = [np.real(d) for d in data[n]]
        y = [np.imag(d) for d in data[n]]

        ax.plot(x, y, 'bo-', linewidth=1)
        line, = ax.plot(solution[0], solution[1], 'ro-', linewidth=1)

        return line,

    create_gif(update, data, len(data), 'trajectory.gif')


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

    create_gif(update, distances, distances.shape[1], 'distance.gif')


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
        plt.scatter(solution[0], solution[1], marker='o', color='r', s=12)
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
    FILE_PATH = "generated_data/serial_test.g2o"

    # Number of vertices in generated data and starting point for test.
    N_VERTICES = 10
    STARTING_VERTEX = 2

    # Create random dataset.
    vertex_list, edge_list = create_random_dataset(0.1, 0.1, N_VERTICES, FILE_PATH)

    solutions = []

    for (vertices, edges) in serial_graph_plan(vertex_list, edge_list, STARTING_VERTEX):
        solutions.append(pgo(vertices, edges)[0])

    create_trajectory_gif(solutions, solutions[-1])
    create_distance_gif(solutions, solutions[-1])

    plot_vertex_trajectory(solutions, solutions[-1])
    plot_pose_graph(vertex_list, edge_list, xlim=[-12, 12], ylim=[-12, 12])
    draw_plots()
