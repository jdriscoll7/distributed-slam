# Add parent directory to python path.
import sys
from os import path
import numpy as np


from solvers.sdp.matrix_creation import w_from_g2o
sys.path.extend([path.dirname(path.abspath(__file__)) + "/../"])

# Import g2o bindings, some custom plotting/parsing functions, and matplotlib.
import g2o
from utility.parsing import parse_g2o
from utility.visualization import plot_vertices, plot_complex_list
from matplotlib import pyplot as plt


# Some simple demo parameters.
#INPUT_FILE = "../solvers/sdp/random_test.g2o"
INPUT_FILE = "/home/joe/repositories/distributed-slam/datasets/input_INTEL_g2o.g2o"
OUTPUT_FILE = "result.g2o"
MAX_ITERATIONS = 200
LOGGING = True


class LocalPGO:

    def __init__(self, input_file):

        self.n = None
        self.optimizer = None
        self.estimate = None

        self.setup_solver(input_file)

    def setup_solver(self, input_file, logging=True):

        solver = g2o.BlockSolverSE2(g2o.LinearSolverPCGSE2())
        solver = g2o.OptimizationAlgorithmGaussNewton(solver)

        # Setup optimizer with optional console logging.
        self.optimizer = g2o.SparseOptimizer()
        self.optimizer.set_algorithm(solver)
        self.optimizer.set_verbose(logging)

        # Load data file and initialize optimizer.
        self.optimizer.load(INPUT_FILE)

        self.n = len(self.optimizer.vertices())
        self.estimate = np.zeros((self.n, 1), dtype=np.complex)

    def optimize_subgraph(self, v):

        for vertex in self.optimizer.vertices():
            if vertex not in v:
                self.optimizer.remove_vertex(vertex)

test = LocalPGO(INPUT_FILE)
test.optimize_subgraph([0, 1, 2, 3, 4, 5, 6, 7, 8])



def run_optimizer():

    # Setup solver.
    solver = g2o.BlockSolverSE2(g2o.LinearSolverPCGSE2())
    solver = g2o.OptimizationAlgorithmGaussNewton(solver)

    # Setup optimizer with optional console logging.
    optimizer = g2o.SparseOptimizer()
    optimizer.set_algorithm(solver)
    optimizer.set_verbose(LOGGING)

    # Load data file and initialize optimizer.
    optimizer.load(INPUT_FILE)
    optimizer.initialize_optimization()

    # Perform optimization and save output.
    optimizer.optimize(MAX_ITERATIONS)
    optimizer.save(OUTPUT_FILE)


def make_plots():

    # Plot result of PGO.
    plot_vertices(parse_g2o(OUTPUT_FILE)[0])
    plt.title("PGO Output (g2o)")

    # Plot original data file.
    plot_vertices(parse_g2o(INPUT_FILE)[0])
    plt.title("Raw Input File")

    # Show both plots.
    plt.show()


def corrected_solution(w, x):

    n = (w.shape[0] + 1) // 2

    L, V = np.linalg.eig(w)

    corrected = np.zeros((x.shape[0], x.shape[1]), dtype=np.complex)
    corrected[0] = x[0]
    corrected[1:] = x[1:] - (V.T[:n]).T @ (V.T[:n]) @ x[1:]



def g2o_to_vector(file_name):

    # Get vertices from g2o.
    v, _ = parse_g2o(file_name)

    # Convert positions and rotations into complex vector.
    positions = np.asarray([x.state[0]+1j*x.state[1] for x in v]).reshape((-1, 1))
    rotations = np.asarray([np.exp(1j*x.state[2]) for x in v]).reshape((-1, 1))

    # Return stacked vector.
    return np.vstack([positions, rotations])

if __name__ == '__main__':

    run_optimizer()
    w = w_from_g2o(INPUT_FILE)

    corrected_solution(w, g2o_to_vector(OUTPUT_FILE))
    # Perform PGO on input file and plot raw data and optimizer solution.



    make_plots()
