import numpy as np
import copy


def _rotation_matrix(theta):

    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


class Vertex:

    def __init__(self, id, position, rotation):
        self.id = id
        self.position = position
        self.rotation = rotation

    def __eq__(self, other):
        return True if other.id == self.id else False

    def set_state(self, position, rotation):

        # Force correct type for rotation.
        if isinstance(rotation, complex):
            rotation = np.angle(rotation)

        if isinstance(position, complex):
            position = np.array([[np.real(position)], [np.imag(position)]])

        if position is not None:
            self.position = position

        if rotation is not None:
            self.rotation = rotation

    def get_connected_edges(self, edges, vertices):
        return get_connected_edges(self.id, edges, vertices)

    def get_complex_position(self):
        return self.position[0] + 1j*self.position[1]

    def get_complex_rotation(self):
        return np.exp(1j*self.rotation)

    def rotate(self, angle, origin=(0, 0)):

        # Precompute needed trigonometric values.
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)

        # Store x and y coordinate of position for brevity.
        x = self.position[0] - origin[0]
        y = self.position[1] - origin[1]

        # Compute new position coordinates.
        new_x = origin[0] + x * cos_angle + y * sin_angle
        new_y = origin[1] + -x * sin_angle + y * cos_angle

        # Update position coordinates.
        self.position[0] = new_x
        self.position[1] = new_y

        # Update angle.
        self.rotation = np.mod(self.rotation + angle, 2 * np.pi)


class Edge:

    def __init__(self, out_vertex, in_vertex, relative_pose, rotation, information_matrix):

        self.out_vertex = out_vertex
        self.in_vertex = in_vertex
        self.relative_pose = relative_pose
        self.rotation = rotation
        self.information_matrix = information_matrix

    def __eq__(self, other):

        if self.in_vertex == other.in_vertex and self.out_vertex == other.out_vertex:
            return True
        else:
            return False


class Graph:

    def __init__(self, vertices, edges):

        self.vertices = [copy.copy(v) for v in vertices]
        self.edges = [copy.copy(e) for e in edges]

    def get_vertices(self):

        return self.vertices

    def get_edges(self):

        return self.edges

    def get_vertex(self, id):

        for v in self.vertices:
            if v.id == id:
                return v

    def get_edge(self, in_vertex, out_vertex):

        for e in self.edges:
            if e.in_vertex == in_vertex and e.out_vertex == out_vertex:
                return e

    def set_state(self, vertex_id, position, rotation):

        # Set depending on the type of position (complex number or vector).
        if isinstance(position, complex):
            position = np.array([[np.real(position)], [np.imag(position)]]).reshape(-1,)

        # Set depending on the type of rotation (complex number or float (radians)).
        if isinstance(rotation, complex):
            rotation = np.angle(rotation)

        self.get_vertex(vertex_id).set_state(position, rotation)

    def update_states(self, state, vertex_ids=None):

        if vertex_ids is None:
            vertex_ids = [i for i in range(state.shape[0])]

        for i in range(state.shape[0] // 2):
            self.set_state(vertex_ids[i], position=state[i, 0], rotation=state[i + state.shape[0] // 2, 0])

    def get_complex_state(self, centered=False):

        x = np.zeros((2*len(self.vertices), 1), dtype=np.complex)

        for i, v in enumerate(self.vertices):
            x[i] = v.position[0] + 1j*v.position[1]
            x[i + len(self.vertices)] = np.exp(1j*v.rotation)

        if centered:
            x[:len(self.vertices)] = x[:len(self.vertices)] - x[0]
            x = x[1:]

        return x

    def subgraph(self, i, reduce=False, neighborhood=False):

        # If i is not a list, make it become a single-element list to allow general neighborhoods.
        if not isinstance(i, list):
            i = [i]

        # Find edges that contain i.
        if neighborhood:
            edges = [copy.copy(e) for e in self.edges if e.out_vertex in i or e.in_vertex in i]
        else:
            edges = [copy.copy(e) for e in self.edges if e.out_vertex in i and e.in_vertex in i]

        # Find vertices covered by edges.
        vertex_ids = list(set([e.in_vertex for e in edges] + [e.out_vertex for e in edges]))
        vertices = [copy.copy(v) for v in self.vertices if v.id in vertex_ids]

        if reduce:
            # Store vertex ids in sorted order and reduce vertex ids.
            vertex_ids = [v.id for v in vertices]
            vertex_ids.sort()

            id_dictionary = {}
            for i, v_id in enumerate(vertex_ids):
                id_dictionary[v_id] = i

            vertices, edges = reduce_ids(vertices, edges)
            return Graph(vertices, edges), id_dictionary

        else:
            return Graph(vertices, edges)

    def remove_edges(self, edge_list):

        for edge in edge_list:
            self.edges.remove(edge)

    def add_edge(self, edge):
        self.edges.append(edge)

    def partition(self, partition_groups=None):

        # Create a copy of graph to remove edges from (prevent unexpected mutations to edges).
        copy_graph = Graph(self.vertices, self.edges)

        graphs = []

        if partition_groups is None:
            partition_groups = [[i for i in range(len(self.vertices))]]

        for vertex_list in partition_groups:

            if vertex_list != partition_groups[-1]:
                # Need an un-reduced and a reduced neighborhood.
                unreduced_neighborhood = copy_graph.subgraph(vertex_list, reduce=False)
                neighborhood, id_list = copy_graph.subgraph(vertex_list, reduce=True)
            else:
                unreduced_neighborhood = copy_graph.subgraph(vertex_list, reduce=False, neighborhood=True)
                neighborhood, id_list = copy_graph.subgraph(vertex_list, reduce=True, neighborhood=True)

            if len(neighborhood.vertices) > 0:
                graphs.append((neighborhood, id_list))

            # Remove edges from previous tree.
            copy_graph.remove_edges(unreduced_neighborhood.edges)

            # Early termination if there are no more edges.
            if len(copy_graph.edges) == 0:
                break

        return graphs

    def neighborhood(self, i):
        """
        Returns all vertices in the neighborhood of vertices in i.

        :param i: List of vertex ids to find neighborhood of.
        :return:  List of vertices that comprise neighborhood of vertex i.
        """

        vertex_list = []

        # Find all vertices with edge connected to vertex i.
        for e in self.edges:
            if e.in_vertex in i and e.out_vertex not in i:
                vertex_list.append(self.get_vertex(e.out_vertex))
            elif e.out_vertex in i and e.in_vertex not in i:
                vertex_list.append(self.get_vertex(e.in_vertex))

        return vertex_list

    def copy(self):
        return Graph(self.get_vertices(), self.get_edges())

    def rotate(self, angle):

        for i, v in enumerate(self.vertices):

            # Extract position and location of vertex.
            position = v.position.reshape((2, 1))
            rotation = v.rotation

            # Compute new rotation and position.
            new_position = (_rotation_matrix(angle) @ position).reshape((2, ))
            new_rotation = rotation - angle

            # Set values.
            self.vertices[i].position = new_position
            self.vertices[i].rotation = new_rotation

    def map_vertex_ids(self, new_ids):

        # Modify vertices first - reorder and then reset ids.
        self.vertices = [self.vertices[i] for i in new_ids]
        for i, v in enumerate(self.vertices):
            self.vertices[i].id = i

        # Modify edges.
        for i, e in enumerate(self.edges):
            self.edges[i].in_vertex = new_ids.index(e.in_vertex)
            self.edges[i].out_vertex = new_ids.index(e.out_vertex)


def reduce_ids(vertices, edges):

    # Copy vertices and edges.
    reduced_v = copy.copy(vertices)
    reduced_e = copy.copy(edges)

    # Maintain a counter for new vertex ids.
    vertex_counter = 0

    # For each vertex id, reduce all edges containing id and then reduce id of vertex.
    for v_index, v in enumerate(vertices):
        for edge_index, e in enumerate(edges):
            if e.in_vertex == v.id:
                reduced_e[edge_index].in_vertex = vertex_counter
            elif e.out_vertex == v.id:
                reduced_e[edge_index].out_vertex = vertex_counter

        # Update id of vertex and increment count.
        reduced_v[v_index].id = vertex_counter
        vertex_counter += 1

    return reduced_v, reduced_e


class MultiGraph:

    def __init__(self, vertex_lists, edge_list):

        # Store vertices by group.
        self.vertex_groups = vertex_lists

        # Lookup table to match vertex id to group.
        self.group_table = [None for i in range(sum([len(v_list) for v_list in vertex_lists]))]

        for index, v_list in enumerate(vertex_lists):
            for v in v_list:
                self.group_table[v.id] = index

        # Get all edges across vertex groups.
        self.cross_edges = [[[] for j in range(len(vertex_lists))] for i in range(len(vertex_lists))]
        self.inner_edges = [[] for i in range(len(vertex_lists))]

        for edge in edge_list:

            # Decide if edge is cross-group or inner-group. Assign accordingly.
            source_group = self.group_table[edge.in_vertex]
            target_group = self.group_table[edge.out_vertex]

            if target_group == source_group:
                self.inner_edges[source_group].append(edge)
            else:
                self.cross_edges[source_group][target_group].append(edge)

    def get_cross_edges(self, group_i, group_j):

        return self.cross_edges[group_i][group_j]

    def get_group_edges(self, group):

        return self.inner_edges[group]

    def get_full_graph(self):

        # Flatten vertex lists.
        vertices = []
        for v_list in self.vertex_groups:
            for v in v_list:
                vertices.append(v)

        # Flatten edge lists.
        edges = []
        for row in self.inner_edges + self.cross_edges:
            for edge_list in row:
                if isinstance(edge_list, list):
                    for edge in edge_list:
                        edges.append(edge)
                else:
                    edges.append(edge_list)

        return vertices, edges, self.group_table


def get_connected_edges(vertex_id, edges, vertices):
    """
    Returns all edges connected to a particular vertex id.

    :param vertex_id:   Vertex to find attached edges of.
    :param edges:       Full list of edges to search.
    :return:            List of edges that have this vertex as in or out vertex.
    """

    # Maintain list of edges that are connect to input vertex.
    connected_edges = []

    # Create list of vertex ids that this edge can be connected to.
    valid_ids = [v.id for v in vertices]

    for edge in edges:
        if edge.in_vertex == vertex_id or edge.out_vertex == vertex_id:
            if edge.in_vertex in valid_ids or edge.out_vertex in valid_ids:
                connected_edges.append(edge)

    return connected_edges
