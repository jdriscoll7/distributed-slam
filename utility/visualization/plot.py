from utility.parsing.g2o import parse_g2o
from matplotlib import pyplot as plt


def plot_vertices(vertices):

    # Extract x coordinates of each vertex.
    x = [v.state[0] for v in vertices]
    y = [v.state[1] for v in vertices]

    # Plot these pairs of coordinates.
    plt.figure()
    plt.plot(x, y, '-')
    plt.axis('off')


if __name__ == "__main__":
    vertices, edges = parse_g2o("../datasets/input_INTEL_g2o.g2o")