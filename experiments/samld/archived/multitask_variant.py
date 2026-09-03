import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

from fraud_gnn.models.archived.multitask import MultiTaskGraphSAGE
from fraud_gnn.models.graphsage_samld import merge_rare_types, load_type_names
from fraud_gnn.utils.metrics import find_best_threshold, report_at_percentile_thresholds
from fraud_gnn.utils.seeding import set_seed
from fraud_gnn.utils.run_logger import log_run

CONFIG_PATH = "configs/samld_binary.yaml"
TYPOLOGY_CONFIG_PATH = "configs/samld_typology.yaml"
S1, S2 = 25, 10
BATCH_SIZE = 256
WEIGHT_CAP = 30.0
TYPE_WEIGHT_CAP = 10.0
EPOCHS = 20
TYPE_LOSS_WEIGHT = 0.5


def sample_k(neighbors, k, rng):
    return neighbors if len(neighbors) <= k else rng.sample(neighbors, k)


def build_batch_subgraph(seed_nodes, adjacency, x, rng, s1=S1, s2=S2):
    hop1_of, all_hop1 = {}, set()
    for s in seed_nodes:
        neighs = sample_k(adjacency.get(s, []), s1, rng)
        hop1_of[s] = neighs
        all_hop1.update(neighs)
    hop2_of, all_hop2 = {}, set()
    for n in all_hop1:
        neighs = sample_k(adjacency.get(n, []), s2, rng)
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


def compute_type_class_weights(y_type_remapped, train_mask, num_classes, cap=TYPE_WEIGHT_CAP):
    illicit_train_types = y_type_remapped[train_mask]
    illicit_train_types = illicit_train_types[illicit_train_types != -1].numpy()
    present_classes = np.unique(illicit_train_types)
    balanced = compute_class_weight("balanced", classes=present_classes, y=illicit_train_types)
    weights = np.ones(num_classes, dtype=np.float32)
    for cls, w in zip(present_classes, balanced):
        weights[cls] = min(w, cap)
    return torch.tensor(weights, dtype=torch.float)


def run_one_seed(seed, x, edge_index, y_binary, y_type_remapped, train_mask, val_mask, test_mask,
                  adjacency, num_type_classes, class_weight_binary, class_weight_type, device,
                  hidden_channels, epochs=EPOCHS):
    set_seed(seed)
    rng = random.Random(seed)

    model = MultiTaskGraphSAGE(in_channels=x.shape[1], hidden_channels=hidden_channels,
                                num_type_classes=num_type_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=5e-4)
    binary_criterion = nn.CrossEntropyLoss(weight=class_weight_binary.to(device))
    type_criterion = nn.CrossEntropyLoss(weight=class_weight_type.to(device), ignore_index=-1)

    train_node_ids = train_mask.nonzero(as_tuple=True)[0].tolist()

    for epoch in range(1, epochs + 1):
        rng.shuffle(train_node_ids)
        model.train()
        for i in range(0, len(train_node_ids), BATCH_SIZE):
            seed_nodes = train_node_ids[i:i + BATCH_SIZE]
            sub_x, sub_edge_index, seed_local_idx = build_batch_subgraph(seed_nodes, adjacency, x, rng)
            sub_x, sub_edge_index = sub_x.to(device), sub_edge_index.to(device)
            seed_local_idx = seed_local_idx.to(device)
            batch_y_binary = y_binary[seed_nodes].to(device)
            batch_y_type = y_type_remapped[seed_nodes].to(device)

            optimizer.zero_grad()
            out_binary, out_type = model(sub_x, sub_edge_index)
            loss_binary = binary_criterion(out_binary[seed_local_idx], batch_y_binary)

            has_valid_type = (batch_y_type != -1).any()
            if has_valid_type:
                loss_type = type_criterion(out_type[seed_local_idx], batch_y_type)
                loss = loss_binary + TYPE_LOSS_WEIGHT * loss_type
            else:
                loss = loss_binary

            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        out_binary, out_type = model(x.to(device), edge_index.to(device))
        probs_binary = F.softmax(out_binary, dim=1)[:, 1].cpu().numpy()
        probs_type = F.softmax(out_type, dim=1).cpu().numpy()
        preds_type = probs_type.argmax(axis=1)

    y_val_b = y_binary[val_mask].numpy()
    probs_val_b = probs_binary[val_mask.numpy()]
    best_t, _ = find_best_threshold(y_val_b, probs_val_b)

    y_test_b = y_binary[test_mask].numpy()
    probs_test_b = probs_binary[test_mask.numpy()]

    y_test_type = y_type_remapped[test_mask].numpy()
    preds_test_type = preds_type[test_mask.numpy()]
    has_type = y_test_type != -1

    type_metrics = {
        "accuracy": accuracy_score(y_test_type[has_type], preds_test_type[has_type]),
        "macro_f1": f1_score(y_test_type[has_type], preds_test_type[has_type], average="macro", zero_division=0),
        "weighted_f1": f1_score(y_test_type[has_type], preds_test_type[has_type], average="weighted", zero_division=0),
    }

    return {
        "seed": seed, "binary_threshold": best_t,
        "probs_test_binary": probs_test_b, "y_test_binary": y_test_b,
        "type_metrics": type_metrics,
    }


def main():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open(TYPOLOGY_CONFIG_PATH, encoding="utf-8") as f:
        typ_cfg = yaml.safe_load(f)

    print("Loading real SAML-D v3 data...")
    data_dict = torch.load(f"{cfg['output_prefix']}.pt", map_location="cpu", weights_only=False)
    x = data_dict["x"]
    edge_index = data_dict["edge_index"]
    y_binary = data_dict["y_binary"]
    y_type = data_dict["y_type"]
    train_mask = data_dict["train_mask"]
    val_mask = data_dict["val_mask"]
    test_mask = data_dict["test_mask"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    remap, num_new_types, _ = merge_rare_types(
        y_type, train_mask, min_train_samples=typ_cfg["min_train_samples_per_type"],
    )
    y_type_remapped = y_type.clone()
    for old_id, new_id in remap.items():
        y_type_remapped[y_type == old_id] = new_id

    n_pos = (y_binary[train_mask] == 1).sum().item()
    n_neg = (y_binary[train_mask] == 0).sum().item()
    class_weight_binary = torch.tensor([1.0, min(n_neg / max(n_pos, 1), WEIGHT_CAP)])
    class_weight_type = compute_type_class_weights(y_type_remapped, train_mask, num_new_types)

    adjacency = defaultdict(list)
    for s, d in zip(edge_index[0].tolist(), edge_index[1].tolist()):
        adjacency[d].append(s)

    all_results = []
    for seed in cfg["seeds"]:
        print(f"\n{'=' * 60}\nSeed {seed}\n{'=' * 60}")
        result = run_one_seed(
            seed, x, edge_index, y_binary, y_type_remapped, train_mask, val_mask, test_mask,
            adjacency, num_new_types, class_weight_binary, class_weight_type, device,
            hidden_channels=cfg["hidden_channels"],
        )
        all_results.append(result)
        print(f"  type macro_f1 this seed: {result['type_metrics']['macro_f1']:.4f}")

        log_run(phase="samld", experiment="archived_multitask_variant", seed=seed,
                metric_name="macro_f1_type", metric_value=result["type_metrics"]["macro_f1"],
                checkpoint_path="", log_path="outputs/logs/samld_archived_multitask.log")

    print(f"\n\n{'=' * 60}\n=== Binary head summary across {len(cfg['seeds'])} seeds ===\n{'=' * 60}")
    for r in all_results:
        print(f"\nseed={r['seed']}:")
        report_at_percentile_thresholds(r["y_test_binary"], r["probs_test_binary"], percentiles=(90, 95, 99))

    print(f"\n\n{'=' * 60}\n=== Type head summary across {len(cfg['seeds'])} seeds ===\n{'=' * 60}")
    for key in ("accuracy", "macro_f1", "weighted_f1"):
        vals = np.array([r["type_metrics"][key] for r in all_results])
        print(f"  {key:12s}: {vals.mean():.4f} +/- {vals.std(ddof=1):.4f}")

    return all_results


if __name__ == "__main__":
    main()
