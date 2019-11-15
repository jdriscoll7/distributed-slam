from solvers.sdp import pgo
from perturbations import delete_random_vertices
from utility.data_generation import create_random_dataset
from utility.parsing import parse_g2o


if __name__ == "__main__":

    # Path for generated file.
    FILE_PATH = "generated_data/abc.g2o"

    # Create random dataset.
    create_random_dataset(0.1, 0.1, 30, FILE_PATH)

    # Read in vertices and edges from created dataset.
    vertices, edges = parse_g2o(FILE_PATH)

    # Run pgo repeatedly, removing one vertex each time until no vertices remain.
    results = {"solutions": [],
               "dual_solutions": [],
               "edges_removed": []}

    while len(vertices) > 1:

        # Run pgo.
        solution, dual_solution = pgo(vertices, edges)

        # Append output to results variable.
        results["solutions"].append(solution)
        results["dual_solutions"].append(dual_solution)

        # Remove single, random vertex.
        vertices, edges, edges_removed = delete_random_vertices(vertices, edges, n=1)

        # Append number of edgees removed to results.
        results["edges_removed"].append(edges_removed)

    print("Finished.")
