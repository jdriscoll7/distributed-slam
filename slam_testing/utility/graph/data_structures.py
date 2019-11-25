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
