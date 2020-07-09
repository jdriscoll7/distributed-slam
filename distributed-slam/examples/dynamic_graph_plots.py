import numpy as np
import matplotlib.pyplot as plt

from solvers.experimental.local_admm import LocalADMM
from utility.visualization import plot_pose_graph
from utility.visualization.plot import plot_graph


def axes_limits(reference_vector):
    
    left = (np.min(reference_vector[0]) if np.min(reference_vector[0]) < 0 else -np.min(reference_vector[0])) - 0.5
    
    right = (-np.max(reference_vector[0]) if np.max(reference_vector[0]) < 0 else np.max(reference_vector[0])) + 0.5
    
    bottom = (np.min(reference_vector[1]) if np.min(reference_vector[1]) < 0 else -np.min(reference_vector[1])) - 0.5
    
    top = (-np.max(reference_vector[1]) if np.max(reference_vector[1]) < 0 else np.max(reference_vector[1])) + 0.5

    return left, right, bottom, top


def make_plots(solution_graphs, colored=False):

    # Axes limits for plotting.
    left, right, bottom, top = axes_limits(solution_graphs[-1].get_complex_state())

    for graph in solution_graphs:

        # Initialize axes.
        plt.figure()
        plt.axes.clear()
        plt.axes.set_xlim(left=left, right=right)
        plt.axes.set_ylim(bottom=bottom, top=top)
        plt.axes.set_xticks([])
        plt.axes.set_yticks([])

        plot_pose_graph(graph=graph, new_figure=False)

        # Save individual frames.
        # plt.savefig('%d.png' % (n))


if __name__ == "__main__":

    admm = LocalADMM(pgo_file="datasets/custom_problem.g2o")

    for i in range(30):
        admm.run_solver(iterations=30, rho=0.1)
        plot_graph(graph=admm.current_estimate())
        plt.show()

    graphs = [x[0] for x in admm.current_estimate().tree_partition()]