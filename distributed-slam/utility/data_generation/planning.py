def serial_graph_plan(vertices, edges, included_vertices=None):
    """
    Creates a plan for adding vertices one-by-one, adding in edges corresponding to added vertex. Function
    is meant to be used as a generator function.

    :param vertices:          Full list of vertices.
    :param edges:             Full list of edges.
    :param included_vertices: List of numbers indicating the sizes at which the plan covers.
    :return:                  Generates vertex and edge lists at each step of the vertex addition.
    """

    # If included_vertices is not supplied, then do all possible sizes.
    if included_vertices is None:
        included_vertices = [i for i in range(1, len(vertices) + 1)]

    # Begin with first vertex.
    current_vertices = [vertices[0]]
    current_edges = []

    for vertex in vertices[1:]:

        # Append current vertex and its neighboring edges.
        current_edges += vertex.get_connected_edges(edges, current_vertices)
        current_vertices += [vertex]

        # Yield current vertex, edge lists if it is at starting vertex or further.
        if len(current_vertices) in included_vertices:
            yield current_vertices, current_edges
