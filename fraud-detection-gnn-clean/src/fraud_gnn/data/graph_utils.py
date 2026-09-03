import random
from collections import defaultdict

import torch


def build_adjacency(edge_index, num_nodes):
    adjacency = defaultdict(list)
    src, dst = edge_index[0].tolist(), edge_index[1].tolist()
    for s, d in zip(src, dst):
        adjacency[s].append(d)
        adjacency[d].append(s)
    return adjacency


def sample_k(neighbors, k):
    return neighbors if len(neighbors) <= k else random.sample(neighbors, k)


def build_batch_subgraph(seed_nodes, adjacency, x, s1=25, s2=10):
    hop1_of, all_hop1 = {}, set()
    for s in seed_nodes:
        neighs = sample_k(adjacency.get(s, []), s1)
        hop1_of[s] = neighs
        all_hop1.update(neighs)

    hop2_of, all_hop2 = {}, set()
    for n in all_hop1:
        neighs = sample_k(adjacency.get(n, []), s2)
        hop2_of[n] = neighs
        all_hop2.update(neighs)

    all_nodes = list(set(seed_nodes) | all_hop1 | all_hop2)
    local_id = {n: i for i, n in enumerate(all_nodes)}

    edges_src, edges_dst = [], []
    for s in seed_nodes:
        for n in hop1_of[s]:
            edges_src.append(local_id[n])
            edges_dst.append(local_id[s])
    for n in all_hop1:
        for n2 in hop2_of[n]:
            edges_src.append(local_id[n2])
            edges_dst.append(local_id[n])

    node_idx_tensor = torch.tensor(all_nodes, dtype=torch.long)
    sub_x = x[node_idx_tensor]
    sub_edge_index = (torch.zeros((2, 0), dtype=torch.long) if not edges_src
                       else torch.tensor([edges_src, edges_dst], dtype=torch.long))
    seed_local_idx = torch.tensor([local_id[s] for s in seed_nodes], dtype=torch.long)
    return sub_x, sub_edge_index, seed_local_idx
