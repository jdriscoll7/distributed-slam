from utility.parsing import parse_g2o
from matplotlib import pyplot as plt
import numpy as np


def plot_complex_list(data):

    # Extract coordinates for each vertex.
    x = [np.real(d) for d in data]
    y = [np.imag(d) for d in data]

    # Plot these pairs of coordinates.
    plt.figure()
    plt.plot(x, y, '-')
    plt.ylim(top=1.1 ** (np.sign(np.max(y))) * np.max(y), bottom=1.1 ** (-1 * np.sign(np.min(y))) * np.min(y))
    plt.xlim(right=1.1 ** (np.sign(np.max(x))) * np.max(x), left=1.1 ** (-1 * np.sign(np.min(x))) * np.min(x))


def plot_vertices(vertices):

    # Extract x coordinates of each vertex.
    x = [v.state[0] for v in vertices]
    y = [v.state[1] for v in vertices]

    # Plot these pairs of coordinates.
    plt.figure()
    plt.plot(x, y, '-')
    plt.ylim(top=1.1**(np.sign(np.max(y)))*np.max(y), bottom=1.1**(-1 * np.sign(np.min(y)))*np.min(y))
    plt.xlim(right=1.1**(np.sign(np.max(x)))*np.max(x), left=1.1**(-1 * np.sign(np.min(x)))*np.min(x))


def draw_plots():

    plt.show()


if __name__ == "__main__":
    vertices, edges = parse_g2o("../datasets/input_INTEL_g2o.g2o")