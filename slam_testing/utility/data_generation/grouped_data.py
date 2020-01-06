from utility.graph.data_structures import MultiGraph


def group_datasets(vertex_lists, edge_lists):

    # Update vertex ids to compensate for overlapping ids.
    base_offsets = [len(v_list) for v_list in vertex_lists]
    offsets = [sum(base_offsets[:i]) for i in range(len(base_offsets))]

    for index, vertex_list in enumerate(vertex_lists):

        offset = offsets[index]

        for v in vertex_list:
            v.id += offset

        for e in edge_lists[index]:
            e.in_vertex += offset
            e.out_vertex += offset

    edge_list = [x for sublist in edge_lists for x in sublist]

    return MultiGraph(vertex_lists=vertex_lists, edge_list=edge_list)
