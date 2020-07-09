import random

from utility.graph import *


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

    # Maintain list of edges removed for each vertex.
    removed_edges = []

    for vertex_id in vertex_ids:

        # Remove individual vertex and increment number of edges removed.
        vertices, edges, removed = remove_vertex(vertices, edges, vertex_id)
        removed_edges.append(removed)

    return vertices, edges, removed_edges


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
    edges_removed = []

    # Remove vertex from vertex list and update ids.
    new_vertices = []

    for v in vertices:
        if v.id != vertex_id:

            # Decrease id by one if its id is greater than the one being removed.
            if v.id > vertex_id:
                v.id -= 1

            # Add this vertex to new vertex list.
            new_vertices.append(v)

    # Initialize new edge list.
    new_edges = []

    # Remove edges containing that vertex.
    for edge in edges:

        # Edge should be removed if the in vertex or out vertex is the target vertex.
        if edge.in_vertex == vertex_id or edge.out_vertex == vertex_id:

            # Increment number of edges removed.
            edges_removed.append(edge)

        else:

            # Update ids.
            edge.in_vertex -= 1 if edge.in_vertex > vertex_id else 0
            edge.out_vertex -= 1 if edge.out_vertex > vertex_id else 0

            # Edge doesn't contain target vertex.
            new_edges.append(edge)

    return new_vertices, new_edges, edges_removed
