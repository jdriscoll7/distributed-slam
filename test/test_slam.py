import g2o
import numpy as np
import os
import visualization.plot_g2o as g2o_plot
from matplotlib import pyplot as plt


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--max_iterations', type=int, default=100000, help='perform n iterations')
parser.add_argument('-i', '--input', type=str, default='../datasets/input_INTEL_g2o.g2o', help='input file')
parser.add_argument('-o', '--output', type=str, default='result', help='save resulting graph as file')
args = parser.parse_args()


def main():
    # solver = g2o.BlockSolverX(g2o.LinearSolverCholmodX())
    solver = g2o.BlockSolverSE2(g2o.LinearSolverEigenSE2())
    solver = g2o.OptimizationAlgorithmLevenberg(solver)

    optimizer = g2o.SparseOptimizer()
    optimizer.set_verbose(True)
    optimizer.set_algorithm(solver)

    optimizer.load(args.input)
    print('num vertices:', len(optimizer.vertices()))
    print('num edges:', len(optimizer.edges()), end='\n\n')

    optimizer.initialize_optimization()
    optimizer.optimize(args.max_iterations)

    if len(args.output) > 0:
        optimizer.save(args.output)


if __name__ == '__main__':
    main()
    g2o_plot.plot_g2o_vertices(g2o_plot.parse_g2o("result")[0])
    g2o_plot.plot_g2o_vertices(g2o_plot.parse_g2o("../datasets/input_INTEL_g2o.g2o")[0])
    plt.show()

