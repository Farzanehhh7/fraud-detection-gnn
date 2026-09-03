import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import f1_score, roc_auc_score, precision_score
from xgboost import XGBClassifier

from fraud_gnn.data.elliptic import load_elliptic
from fraud_gnn.models.archived.hybrid_and_skipgcn import EmbeddingExtractor

CONFIG_PATH = "configs/elliptic_binary.yaml"
EMBEDDING_DIM = 64
EPOCHS = 100
SEED = 42


def main():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = load_elliptic(
        features_path=cfg["features_path"], edges_path=cfg["edges_path"], classes_path=cfg["classes_path"],
        train_end=cfg["train_end"], val_end=cfg["val_end"], device=device,
    )
    x, y, edge_index = data["x"], data["y"], data["edge_index"]
    train_mask, test_mask = data["train_mask"], data["test_mask"]
    x_raw = x.cpu().numpy()

    emb_model = EmbeddingExtractor(in_channels=x.shape[1], hidden_channels=cfg["hidden_channels"],
                                    embedding_dim=EMBEDDING_DIM).to(device)
    optimizer = torch.optim.Adam(emb_model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    emb_model.train()
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        out = emb_model(x, edge_index)
        loss = criterion(out[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

    emb_model.eval()
    with torch.no_grad():
        graph_embeddings = emb_model(x, edge_index).cpu().numpy()

    X_combined = np.hstack([x_raw, graph_embeddings])
    X_train = X_combined[train_mask.cpu().numpy()]
    y_train = y[train_mask].cpu().numpy()
    X_test = X_combined[test_mask.cpu().numpy()]
    y_test = y[test_mask].cpu().numpy()

    n_neg_train, n_pos_train = (y_train == 0).sum(), (y_train == 1).sum()
    xgb_hybrid = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        scale_pos_weight=float(n_neg_train / n_pos_train), eval_metric="logloss",
    )
    xgb_hybrid.fit(X_train, y_train)

    y_pred = xgb_hybrid.predict(X_test)
    y_prob = xgb_hybrid.predict_proba(X_test)[:, 1]

    print(f"F1 illicit: {f1_score(y_test, y_pred):.4f}")
    print(f"AUC: {roc_auc_score(y_test, y_prob):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")

    return {"F1": f1_score(y_test, y_pred), "AUC": roc_auc_score(y_test, y_prob)}


if __name__ == "__main__":
    main()
