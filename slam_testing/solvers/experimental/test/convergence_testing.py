from solvers.experimental.local_admm import LocalADMM
from utility.graph import Graph
from utility.parsing.g2o import write_g2o, parse_g2o
from utility.data_generation.planning import serial_graph_plan


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

if __name__ == "__main__":

    # Generate incremental graphs.
    graph_list = incremental_graphs(pgo_file="/home/joe/repositories/distributed-slam/datasets/custom_problem.g2o")

    # Store history of local variables.
    local_variables = []

    # Initialize optimizer to first of graphs.
    initial_graph = graph_list[0]
    admm_optimizer = LocalADMM(graph=initial_graph)

    # Solve and append local variables from initial graph.
    admm_optimizer.run_solver(iterations=30)
    local_variables.append(admm_optimizer.local_variables)

    for g in graph_list[1:]:

        # "Augment" the problem (i.e. add vertex by creating one new local variable without changing other subproblems).
        admm_optimizer.augment(g, vertex_id=g.vertices[-1].id)

        # Run solver.
        admm_optimizer.run_solver(iterations=30)

        # Append the local variables to the history of local variables.
        local_variables.append(admm_optimizer.local_variables)


    print(local_variables)
