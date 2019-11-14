import random


def delete_random_vertices(vertices, edges, n):
    """
    Deletes n random vertices from vertex list and removes the edges corresponding to them.

    :param vertices:    List of Vertex objects
    :param edges:       List of Edge objects
    :param n:           Number of vertices to remove
    :return:            (new_vertices, new_edges, edges_removed) - edges_removed is the number of edges that were
                        deleted in the end
    """

    # Make sure the number of vertices being removed makes sense.
    assert(n <= len(vertices))

    # Shuffle list of vertices and pick ids of first n vertices for removal.
    random.shuffle(vertices)
    removal_ids = [vertices[i].id for i in range(n)]

    # Remove these and return result.
    return remove_vertices(vertices, edges, removal_ids)


def remove_vertices(vertices, edges, vertex_ids):
    """
    Wrapper for remove_vertex function designed to remove multiple vertices at once.

    :param vertices:    List of Vertex objects
    :param edges:       List of Edge objects
    :param vertex_ids:  ID numbers of vertices to remove
    :return:            (new_vertices, new_edges, edges_removed) - edges_removed is the number of edges that were
                        deleted in the end
    """

    # Need separate details variable to sum up results of individual removals.
    edges_removed = 0

    for vertex_id in vertex_ids:

        # Remove individual vertex and increment number of edges removed.
        vertices, edges, n_removed = remove_vertex(vertices, edges, vertex_id)
        edges_removed += n_removed

    return vertices, edges, edges_removed


def remove_vertex(vertices, edges, vertex_id):
    """
    Removes a single vertex from vertex and edge list pair. Deletes vertex from vertex list and removes all edges
    containing that vertex from the edge list.

    :param vertices:    List of Vertex objects
    :param edges:       List of Edge objects
    :param vertex_id:   ID number of vertex to remove
    :return:            (new_vertices, new_edges, edges_removed) - edges_removed is the number of edges that were
                        deleted in the end
    """

    # Initialize output details.
    edges_removed = 0

    # Remove vertex from vertex list.
    new_vertices = [v for v in vertices if v.id != vertex_id]

    # Initialize new edge list.
    new_edges = []

    # Remove edges containing that vertex.
    for edge in edges:

        # Edge should be removed if the in vertex or out vertex is the target vertex.
        if edge.in_vertex == vertex_id or edge.out_vertex == vertex_id:

            # Increment number of edges removed.
            edges_removed += 1

        else:

            # Edge doesn't contain target vertex - add to new edge list.
            new_edges.append(edge)

    return new_vertices, new_edges, edges_removed
