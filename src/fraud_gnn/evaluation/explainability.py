import numpy as np
import torch.nn.functional as F


def explain_decision(model, x, edge_index, node_idx, feature_names, y, device,
                      top_k_features=5, top_k_neighbors=5, verbose=True):
    model.eval()
    x_grad = x.clone().to(device).requires_grad_(True)
    out = model(x_grad, edge_index.to(device))
    probs = F.softmax(out, dim=1)[:, 1]
    illicit_logit = out[node_idx, 1]

    illicit_logit.backward()
    grad = x_grad.grad

    own_grad = grad[node_idx].detach().cpu().numpy()
    own_features = x[node_idx].detach().cpu().numpy()
    own_attribution = own_grad * own_features
    order = np.argsort(-np.abs(own_attribution))

    label_val = y[node_idx].item()
    label = "illicit" if label_val == 1 else ("licit" if label_val == 0 else "unknown")

    if verbose:
        print(f"\n{'=' * 70}\nnode {node_idx}   P(illicit)={probs[node_idx].item():.4f}   "
              f"actual label={label}\n{'=' * 70}")
        for i in order[:top_k_features]:
            print(f"  {feature_names[i]:28s} value={own_features[i]:+.3f}   "
                  f"attribution={own_attribution[i]:+.4f}")

    grad_magnitude = grad.abs().sum(dim=1).cpu().numpy()
    grad_magnitude[node_idx] = 0.0
    n_neighborhood = int((grad_magnitude > 0).sum())
    top_neighbors = np.argsort(-grad_magnitude)[:top_k_neighbors]
    top_neighbors = [n for n in top_neighbors if grad_magnitude[n] > 0]

    if verbose:
        print(f"\n{n_neighborhood} nodes in the 2-hop computational neighborhood:")
        for n_idx in top_neighbors:
            n_label_val = y[n_idx].item()
            n_label = "illicit" if n_label_val == 1 else ("licit" if n_label_val == 0 else "unknown")
            print(f"  node {n_idx:8d}   influence={grad_magnitude[n_idx]:.4f}   actual label={n_label}")

    return {
        "node_idx": node_idx,
        "prob_illicit": probs[node_idx].item(),
        "own_attribution": dict(zip(feature_names, own_attribution.tolist())),
        "n_neighborhood": n_neighborhood,
        "top_neighbors": [(int(n), float(grad_magnitude[n]), int(y[n].item())) for n in top_neighbors],
    }
