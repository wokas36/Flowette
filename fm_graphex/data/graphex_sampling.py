import numpy as np
import igraph as ig
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def sample_graph_from_graphon(graphon, n):
    """
    Generate a graph with n nodes from a graphon (m x m matrix)
    Parameters:
    -----------
    graphon : numpy array of shape (m, m)
    The graphon matrix with values in [0, 1]
    n : int
    Number of nodes in the generated graph
    Returns:
    --------
    adj_matrix : numpy array of shape (n, n)
    Adjacency matrix of the generated graph
    """
    m = graphon.shape[0]
    # Sample n uniform points in [0, 1]
    u = np.random.uniform(0, 1, n)
    # Map each point to graphon indices
    indices = (u * m).astype(int)
    indices = np.clip(indices, 0, m-1) # Ensure within bounds
    # Initialize adjacency matrix
    adj_matrix = np.zeros((n, n))
    # Generate edges based on graphon probabilities
    for i in range(n):
        for j in range(i+1, n):
            # Get probability from graphon
            prob = graphon[indices[i], indices[j]]
            # Generate edge with that probability
            if np.random.rand() < prob:
                adj_matrix[i, j] = 1
                adj_matrix[j, i] = 1 # Undirected graph
    return adj_matrix


def sample_graphon_from_sparsified_graphon(graphon, n, sparsity_factor, epsilon=1e-3):
    """
    Generate a sparse graphon from a given graphon by applying a sparsity factor.
    Parameters:
    -----------
    graphon : numpy array of shape (m, m)
    The original graphon matrix with values in [0, 1]
    n : int
    Number of nodes in the generated graph
    sparsity_factor : float
    Factor to scale down the graphon values to induce sparsity (0 < sparsity_factor <= 1)

    Returns:
    --------
    sparse_graphon : numpy array of shape (m, m)
    The sparse graphon matrix
    """
    meanW = np.mean(graphon)

    if sparsity_factor is None:
        sparsity_factor = 1/(meanW*n) + epsilon

    # Scale down the graphon values
    sparse_graphon = graphon * sparsity_factor

    # Sample graph from sparse graphon
    adj_matrix = sample_graph_from_graphon(sparse_graphon, n)
    
    # pdb.set_trace()
    return adj_matrix


def add_disjoint_ring_components(gr, ring_size_counts, max_nodes): # add_specific_rings_bounded
    """
    Add specific rings with a node budget.
    Add rings of specific sizes matching QM9 statistics, but never exceed max_nodes.
    ring_size_counts = {5:2, 6:1} means:
       add up to two 5-member rings and one 6-member ring, subject to the node budget.
    """

    result = gr.copy()

    for ring_size, count in ring_size_counts.items():
        for _ in range(count):
            remaining = max_nodes - result.vcount()
            if remaining <= 0:
                # No capacity left: stop adding rings entirely
                return result

            # If we don't have enough budget for a full ring, shrink the ring size
            # or skip if too small to be meaningful (e.g., < 3 nodes).
            ring_size_eff = min(ring_size, remaining)
            if ring_size_eff < 3:
                # Not enough room for a reasonable ring, skip this one
                continue

            ring = ig.Graph.Ring(ring_size_eff)
            current_n = result.vcount()
            result = result.disjoint_union(ring)

            # random connection to original graph
            orig_node = np.random.randint(0, current_n)
            ring_node = np.random.randint(current_n, current_n + ring_size_eff)
            result.add_edge(orig_node, ring_node)

    return result


def add_path_based_ring_closures(gr, ring_size_dict, max_nodes): # add_embedded_rings_bounded

    if not isinstance(gr, ig.Graph):
        raise ValueError("Input must be an igraph.Graph")

    result_graph = gr.copy()

    for ring_size, count in ring_size_dict.items():
        for _ in range(count):
            remaining = max_nodes - result_graph.vcount()
            if remaining <= 0:
                return result_graph

            n_vertices = result_graph.vcount()
            # If we really don't have at least 1 node yet, just create a ring component
            if n_vertices == 0:
                nodes_to_add = min(ring_size, remaining)
                if nodes_to_add < 3:
                    # not enough room for a meaningful ring
                    return result_graph
                current_n = result_graph.vcount()
                result_graph.add_vertices(nodes_to_add)
                new_nodes = list(range(current_n, current_n + nodes_to_add))
                ring_edges = [
                    (new_nodes[i], new_nodes[(i + 1) % nodes_to_add])
                    for i in range(nodes_to_add)
                ]
                result_graph.add_edges(ring_edges)
                # skip embedding for this ring, go to next
                continue

            # normal path-finding logic here
            path_found = False
            attempts = 0
            max_attempts = 100
            path_length = ring_size - 1

            while not path_found and attempts < max_attempts:
                attempts += 1

                start_vertex = np.random.randint(0, n_vertices)

                for tlen in range(2, min(path_length, n_vertices - 1) + 1):
                    all_vertices = list(range(n_vertices))
                    potential_ends = [v for v in all_vertices if v != start_vertex]

                    if len(potential_ends) == 0:
                        continue

                    potential_ends = np.random.choice(
                        potential_ends,
                        size=min(10, len(potential_ends)),
                        replace=False,
                    )

                    for end_vertex in potential_ends:
                        paths = result_graph.get_shortest_paths(
                            start_vertex, to=end_vertex, mode="all"
                        )
                        if paths and len(paths[0]) > 1:
                            path = paths[0]
                            if 2 <= len(path) <= path_length:
                                existing_path = path
                                path_found = True
                                break

                    if path_found:
                        break

            # Fallback when no path is found
            if not path_found:
                edges = result_graph.get_edgelist()

                if len(edges) > 0:
                    selected_edge = edges[np.random.randint(len(edges))]
                    existing_path = list(selected_edge)
                elif n_vertices >= 2:
                    # safe: we have at least 2 nodes
                    existing_path = np.random.choice(
                        n_vertices, 2, replace=False
                    ).tolist()
                else:
                    # n_vertices == 1: we can only anchor at that single vertex
                    existing_path = [0]

            # Add new nodes + edges
            desired_nodes_to_add = max(0, ring_size - len(existing_path))
            remaining = max_nodes - result_graph.vcount()
            nodes_to_add = min(desired_nodes_to_add, remaining)

            current_n = result_graph.vcount()
            edges_to_add = []

            if nodes_to_add > 0:
                result_graph.add_vertices(nodes_to_add)
                new_nodes = list(range(current_n, current_n + nodes_to_add))

                # connect start -> new_nodes -> end
                edges_to_add.append((existing_path[0], new_nodes[0]))
                for j in range(nodes_to_add - 1):
                    edges_to_add.append((new_nodes[j], new_nodes[j + 1]))
                edges_to_add.append((new_nodes[-1], existing_path[-1]))
            else:
                # no budget for new nodes, just close the ring if possible
                if len(existing_path) >= 2 and not result_graph.are_connected(
                    existing_path[0], existing_path[-1]
                ):
                    edges_to_add.append((existing_path[0], existing_path[-1]))

            if edges_to_add:
                result_graph.add_edges(edges_to_add)

    return result_graph


def generate_graph_from_graphex(
    graphon,
    n,
    sparsity_factor,
    epsilon=1e-3,
    ring_proportion=0.1,
    ring_sizes={5: 1, 6: 1},
    embedproportion=0.1
):
    """
    Generate a graph from a graphex by sampling a sparse graphon and adding rings,
    ensuring the total number of nodes is <= n.
    """
    # Sample sparse graph from graphex
    adj_matrix = sample_graphon_from_sparsified_graphon(
        graphon, n, sparsity_factor=sparsity_factor, epsilon=epsilon
    )

    # Create igraph object from adjacency matrix
    gr = ig.Graph.Adjacency(adj_matrix.tolist(), mode="undirected")

    # Keep the largest connected component
    components = gr.connected_components()
    largest_cc = max(components, key=len)
    gr = gr.subgraph(largest_cc)

    # If largest CC itself is too big, trim it to n nodes (induced subgraph).
    if gr.vcount() > n:
        keep = np.random.choice(gr.vcount(), size=n, replace=False)
        gr = gr.subgraph(keep.tolist())

    # If no rings requested, just return (already <= n)
    if not ring_sizes:
        return gr

    # Add specific rings under budget
    gr1 = add_disjoint_ring_components(gr, ring_sizes, max_nodes=n)

    # Add embedded rings under budget
    final_graph = add_path_based_ring_closures(gr1, ring_sizes, max_nodes=n)

    # Final safety check (should already hold)
    if final_graph.vcount() > n:
        # As a hard fallback, induce a subgraph on n nodes
        keep = np.random.choice(final_graph.vcount(), size=n, replace=False)
        final_graph = final_graph.subgraph(keep.tolist())

    return final_graph