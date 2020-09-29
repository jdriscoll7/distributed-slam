# Add parent directory to python path.
import sys
from os import path
sys.path.extend([path.dirname(path.abspath(__file__)) + "/../"])

# Import g2o bindings, some custom plotting/parsing functions, and matplotlib.
import g2o
from utility.parsing import parse_g2o
from utility.visualization import plot_vertices
from matplotlib import pyplot as plt


# Some simple demo parameters.
#INPUT_FILE = "../solvers/sdp/random_test.g2o"
INPUT_FILE = "/home/joe/repositories/distributed-slam/datasets/input_INTEL_g2o.g2o"
OUTPUT_FILE = "misc/result.g2o"
MAX_ITERATIONS = 100
LOGGING = True


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


if __name__ == '__main__':

    # Perform PGO on input file and plot raw data and optimizer solution.
    run_optimizer()
    make_plots()
