import matplotlib.pyplot as plt


from solvers.sdp import pgo
from utility.visualization import plot_pose_graph, plot_vertices, draw_plots
from utility.parsing import parse_g2o


class Measurements:

    def __init__(self, vertices=[], edges=[], file_name=None):

        if file_name != None:
            vertices, edges = parse_g2o(file_name)

        # Store measurements - vertices and relative measurements between them make up PGO problem.
        self.vertices = vertices
        self.edges = edges

        # Store the rotation of the first vertex (anchoring).
        if len(vertices) > 0:
            self.anchor_rotation = -vertices[0].state[2]
            self.anchor_position = (vertices[0].state[0],
                                    vertices[0].state[1])

    def __add__(self, measurement):

        # Append vertex and edge lists.
        vertices = self.get_vertices() + measurement.get_vertices()
        edges = self.get_edges() + measurement.get_edges()

        return Measurements(vertices, edges)

    def __len__(self):

        return len(self.vertices)

    def update(self, positions, rotations):

        for n in range(len(rotations)):
            self.vertices[n].set_state(positions[n][0], rotations[n][0])

    def get_rotated_vertices(self):

        # Copy vertices for use in new measurements object.
        new_measurements = Measurements(vertices=self.vertices.copy(), edges=self.edges.copy())

        # Rotate vertices.
        new_measurements.rotate_vertices(self.anchor_rotation, self.anchor_position)

        return new_measurements

    def rotate_vertices(self, angle, origin=(0, 0)):

        for n in range(len(self.vertices)):
            self.vertices[n].rotate(angle, origin)

    def get_vertices(self):

        return self.vertices

    def get_edges(self):

        return self.edges


class Agent:

    def __init__(self, self_measurements=None, shared_measurements=None):

        # Initialize some class variables.
        if isinstance(self_measurements, str):
            self.self_measurements = Measurements(file_name=self_measurements)
        else:
            self.self_measurements = Measurements(self_measurements)

        self.shared_measurements = Measurements()

        # Store flag that indicates if PGO has been run yet. Determines when to use anchor rotation in plotting.
        self.pgo_run = False

    def add_measurements(self, measurements):

        self.measurements += measurements

    def update_positions(self, new_positions, new_rotations):

        # First self measurements.
        self.self_measurements.update(new_positions[:len(self.self_measurements)],
                                      new_rotations[:len(self.self_measurements)])

        # Update shared measurements.
        self.shared_measurements.update(new_positions[:len(self.shared_measurements)],
                                        new_rotations[:len(self.shared_measurements)])

    def run_pgo(self):

        # Combine shared measurements with self measurements.
        measurements = self.self_measurements + self.shared_measurements

        # Run pgo and get results.
        positions, rotations, dual_solution = pgo(measurements.get_vertices(), measurements.get_edges())

        # Update agent's positions.
        self.update_positions(positions, rotations)

        # Update flag to indicate pgo has been run.
        self.pgo_run = True

    def plot_current_estimate(self):

        # Get current position estimates.
        if self.pgo_run:
            measurements = self.self_measurements#.get_rotated_vertices()
        else:
            measurements = self.self_measurements

        vertices = measurements.get_vertices()
        edges = measurements.get_edges()

        # Make plots.
        plot_pose_graph(vertices, edges)
        plot_vertices(vertices)
        draw_plots()


if __name__ == "__main__":

    agent = Agent("/home/joe/repositories/distributed-slam/distributed-slam/utility/data_generation/test/dataset.g2o")
    #agent = Agent("/home/joe/repositories/distributed-slam/datasets/input_INTEL_g2o.g2o")
    agent.plot_current_estimate()
    agent.run_pgo()
    agent.plot_current_estimate()
    agent.run_pgo()