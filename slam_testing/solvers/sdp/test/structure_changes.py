from solvers.sdp import w_from_vertices_and_edges
from utility.data_generation import create_random_dataset
from utility.parsing import parse_g2o


import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":

    v, e = create_random_dataset(0.1, 0.1, 30, 'data/random_test.g2o')
    #v, e = parse_g2o('data/random_test.g2o')

    # Shuffle edges.
    np.random.shuffle(e)

    # Print information about edge being removed.
    print("Removed edge information:")
    print("\t - edge_number: \t\t%d" % (len(e) - 1))
    print("\t - in-vertex: \t\t\t%d" % (e[-1].in_vertex))
    print("\t - out-vertex: \t\t\t%d" % (e[-1].out_vertex))
    print("\t - rotation: \t\t\t%f" % (e[-1].rotation))
    print("\t - relative pose x: \t%f" % (e[-1].relative_pose[0]))
    print("\t - relative pose y: \t%f\n" % (e[-1].relative_pose[1]))

    # Compute changed indices.
    diag_1 = e[-1].in_vertex - 1
    diag_2 = e[-1].out_vertex - 1
    diag_3 = e[-1].in_vertex + len(v) - 1
    diag_4 = e[-1].out_vertex + len(v) - 1

    # Print predictions.
    print("Predicted changes")
    print("\t - [%d, %d]" % (diag_1, diag_1))
    print("\t - [%d, %d]" % (diag_1, diag_2))
    print("\t - [%d, %d]" % (diag_1, diag_3))
    print("\t - [%d, %d]" % (diag_2, diag_1))
    print("\t - [%d, %d]" % (diag_2, diag_2))
    print("\t - [%d, %d]" % (diag_2, diag_4))
    print("\t - [%d, %d]" % (diag_3, diag_3))
    print("\t - [%d, %d]" % (diag_3, diag_4))
    print("\t - [%d, %d]" % (diag_4, diag_1))
    print("\t - [%d, %d]" % (diag_4, diag_2))
    print("\t - [%d, %d]" % (diag_4, diag_3))
    print("\t - [%d, %d]" % (diag_4, diag_4))

    w = np.asarray(w_from_vertices_and_edges(v, e[:-1]))
    next_w = np.asarray(w_from_vertices_and_edges(v, e))

    print("Changed entries (%d):" % (np.shape(np.argwhere(w != next_w))[0]))
    print("Before and after:")
    print(np.array(np.hstack([w[w != next_w].reshape(-1, 1), next_w[w != next_w].reshape(-1, 1)])))
    print("After:")
    print()

    print(np.argwhere(w != next_w))

    fig, ax = plt.subplots()

    im = plt.spy(w != next_w)
    plt.show()