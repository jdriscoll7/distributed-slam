import numpy as np

from examples.kite_test import kite
from solvers.admm.local_admm import LocalADMM
from utility.common import cost_function
from utility.graph import Graph
from utility.parsing.g2o import write_g2o, parse_g2o
from utility.data_generation.planning import serial_graph_plan
from solvers.sdp import pgo
from utility.visualization import plot_pose_graph


def incremental_graphs(pgo_file, directory='./serial_graphs/'):

    # Parse g2o file.
    vertices, edges = parse_g2o(path=pgo_file)

    # Return list of incremental graphs.
    graph_list = []

    # Create and write incremental graphs.
    for v, e in serial_graph_plan(vertices, edges):
        write_g2o(v, e, directory + str(len(v)) + '.g2o')
        graph_list.append(Graph(v, e))

    return graph_list


def print_local_variables(local_variables):

    np.set_printoptions(linewidth=np.inf)

    for i in range(len(local_variables)):
        print("History for local variable %d: \n" % (i))
        for x in local_variables[i:]:
            print(x[i])
            print("\n")
        print("\n\n")


def print_eigenvector_history(local_variables):

    np.set_printoptions(linewidth=np.inf)
    for i in range(len(local_variables)):
        print("Eigenvectors for local variable %d: \n" % (i))
        for x in local_variables[i:]:
            eigenvectors = np.linalg.eigh(x[i])[1]
            print(eigenvectors)
            print("\n")
        print("\n\n")


def global_solution_graph(file_name):

    return pgo(graph=Graph(*parse_g2o(file_name)), file_name=file_name)[0]


if __name__ == "__main__":


    kite(n=8, d=2, save_file="/home/joe/repositories/distributed-slam/datasets/kite_8_2.g2o")
    PGO_FILE = "/home/joe/repositories/distributed-slam/datasets/kite_8_2.g2o"

    # Find global solution.
    global_solution = global_solution_graph(PGO_FILE)

    # cost_function(global_solution)

    # Store history of local variables.
    local_variables = []

    # Initialize optimizer to first of graphs.
    admm_optimizer = LocalADMM(graph=global_solution, partition=[[0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10]])
    admm_optimizer.current_estimate()

    # Solve and append local variables from initial graph.
    # admm_optimizer.run_solver(iterations=100)
    # local_variables.append(admm_optimizer.local_variables)

    for i in range(1000):

        # Run solver.
        admm_optimizer.run_solver(iterations=100, rho=0.1)

        # Append the local variables to the history of local variables.
        local_variables.append(admm_optimizer.local_variables)
        print("Step %d - Cost: %f" % (i + 3, cost_function(admm_optimizer.current_estimate())))

    print_local_variables(local_variables)
    print_eigenvector_history(local_variables)
