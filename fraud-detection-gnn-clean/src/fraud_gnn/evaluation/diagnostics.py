from collections import defaultdict

import numpy as np


def compute_homophily(df_edge, label_map):
    src_labels = df_edge["txId1"].map(label_map)
    dst_labels = df_edge["txId2"].map(label_map)
    both_labeled = (src_labels != -1) & (dst_labels != -1)

    n_both_labeled = int(both_labeled.sum())
    n_same = int(((src_labels == dst_labels) & both_labeled).sum())
    edge_homophily = n_same / n_both_labeled

    illicit_pairs = int(((src_labels == 1) & (dst_labels == 1) & both_labeled).sum())
    licit_pairs = int(((src_labels == 0) & (dst_labels == 0) & both_labeled).sum())
    mixed_pairs = n_both_labeled - illicit_pairs - licit_pairs

    neighbors = defaultdict(set)
    for src, dst, ls, ld in zip(df_edge["txId1"], df_edge["txId2"], src_labels, dst_labels):
        if ls != -1 and ld != -1:
            neighbors[src].add(dst)
            neighbors[dst].add(src)

    node_scores_all, node_scores_illicit, node_scores_licit = [], [], []
    for node, neighs in neighbors.items():
        own_label = label_map.get(node, -1)
        if own_label == -1:
            continue
        neigh_labels = [label_map.get(n, -1) for n in neighs]
        neigh_labels = [l for l in neigh_labels if l != -1]
        if not neigh_labels:
            continue
        score = np.mean([1 if l == own_label else 0 for l in neigh_labels])
        node_scores_all.append(score)
        (node_scores_illicit if own_label == 1 else node_scores_licit).append(score)

    return {
        "n_both_labeled": n_both_labeled,
        "edge_homophily": edge_homophily,
        "illicit_pairs": illicit_pairs,
        "licit_pairs": licit_pairs,
        "mixed_pairs": mixed_pairs,
        "node_homophily_all": float(np.mean(node_scores_all)),
        "node_homophily_illicit": float(np.mean(node_scores_illicit)),
        "node_homophily_licit": float(np.mean(node_scores_licit)),
        "n_illicit_nodes": len(node_scores_illicit),
        "n_licit_nodes": len(node_scores_licit),
    }


def check_edge_temporal_leakage(edge_index_cpu, time_steps_raw, train_end):
    src_ts = time_steps_raw[edge_index_cpu[0]]
    dst_ts = time_steps_raw[edge_index_cpu[1]]

    forward = (src_ts <= dst_ts).sum().item()
    backward = (src_ts > dst_ts).sum().item()
    total = len(src_ts)

    dst_in_train = dst_ts <= train_end
    src_ahead_of_dst = src_ts > dst_ts
    leaking_edges = (dst_in_train & src_ahead_of_dst).sum().item()

    forward_edge_mask = src_ts <= dst_ts
    edge_index_masked = edge_index_cpu[:, forward_edge_mask]

    return {
        "total_edges": total,
        "forward_edges": forward,
        "backward_edges": backward,
        "leaking_edges": leaking_edges,
        "edge_index_masked": edge_index_masked,
        "n_edges_after_masking": edge_index_masked.shape[1],
    }
