class Vertex:

    def __init__(self, id, state):
        self.id = id
        self.state = state

    def get_connected_edges(self, edges, vertices):
        return get_connected_edges(self.id, edges, vertices)


class Edge:

    def __init__(self, out_vertex, in_vertex, relative_pose, rotation, information_matrix):
        self.out_vertex = out_vertex
        self.in_vertex = in_vertex
        self.relative_pose = relative_pose
        self.rotation = rotation
        self.information_matrix = information_matrix


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
