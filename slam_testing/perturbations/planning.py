def serial_graph_plan(vertices, edges):
    """
    Creates a plan for adding vertices one-by-one, adding in edges corresponding to added vertex. Function
    is meant to be used as a generator function.

    :param vertices:    Full list of vertices.
    :param edges:       Full list of edges.
    :return:            Generates vertex and edge lists at each step of the vertex addition.
    """

    # Output lists that will be appended to during execution.
    current_vertices, current_edges = [], []

    for vertex in vertices:

        # Append current vertex and its neighboring edges.
        current_vertices += vertex
        current_edges += vertex.get_connected_edges(edges)

        # Yield current vertex, edge lists.
        yield current_vertices, current_edges
