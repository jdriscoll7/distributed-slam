class Vertex:

    def __init__(self, id, state):
        self.id = id
        self.state = state

    def get_connected_edges(self, edges):
        return get_connected_edges(self.id, edges)


class Edge:

    def __init__(self, out_vertex, in_vertex, relative_pose, rotation, information_matrix):
        self.out_vertex = out_vertex
        self.in_vertex = in_vertex
        self.relative_pose = relative_pose
        self.rotation = rotation
        self.information_matrix = information_matrix


def get_connected_edges(vertex_id, edges):
    """
    Returns all edges connected to a particular vertex id.

    :param vertex_id:   Vertex to find attached edges of.
    :param edges:       Full list of edges to search.
    :return:            List of edges that have this vertex as in or out vertex.
    """

    connected_edges = []

    for edge in edges:
        if edge.in_vertex == vertex_id or edge.out_vertex == vertex_id:
            connected_edges.append(edge)

    return connected_edges
