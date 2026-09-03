import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from fraud_gnn.data.elliptic import load_elliptic
from fraud_gnn.models.archived.attention_variants import ATGATModulated
from fraud_gnn.utils.metrics import evaluate_binary, find_best_threshold, run_multi_seed

CONFIG_PATH = "configs/elliptic_binary.yaml"
EPOCHS = 200
TEMPORAL_DIM = 32
SEEDS = (42, 1, 7, 123, 2024)


def main():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_elliptic(
        features_path=cfg["features_path"], edges_path=cfg["edges_path"], classes_path=cfg["classes_path"],
        train_end=cfg["train_end"], val_end=cfg["val_end"], device=device,
    )
    x, y, edge_index = data["x"], data["y"], data["edge_index"]
    train_mask, val_mask, test_mask = data["train_mask"], data["val_mask"], data["test_mask"]
    class_weights = data["class_weights"]
    time_steps = (data["time_steps_raw"] - 1).to(device)
    num_timesteps = int(time_steps.max().item()) + 1

    def run_one_seed(seed):
        torch.manual_seed(seed)
        model = ATGATModulated(in_channels=x.shape[1], hidden_channels=cfg["hidden_channels"], out_channels=2,
                                num_timesteps=num_timesteps, temporal_dim=TEMPORAL_DIM).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        for epoch in range(1, EPOCHS + 1):
            model.train()
            optimizer.zero_grad()
            out = model(x, edge_index, time_steps)
            loss = criterion(out[train_mask], y[train_mask])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            out = model(x, edge_index, time_steps)
            probs_all = F.softmax(out, dim=1)[:, 1].cpu().numpy()

        y_val = y[val_mask].cpu().numpy()
        probs_val = probs_all[val_mask.cpu().numpy()]
        best_t, _ = find_best_threshold(y_val, probs_val)

        y_test = y[test_mask].cpu().numpy()
        probs_test = probs_all[test_mask.cpu().numpy()]
        preds_test = (probs_test >= best_t).astype(int)
        return evaluate_binary("FiLM modulated", y_test, preds_test, probs_test, verbose=False)

    return run_multi_seed(run_one_seed, seeds=SEEDS, name="FiLM + global")


if __name__ == "__main__":
    main()
