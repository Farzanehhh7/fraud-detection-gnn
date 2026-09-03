import csv
import datetime
import os
import subprocess

REGISTRY_PATH = "outputs/run_registry.csv"
FIELDS = [
    "run_id", "timestamp", "phase", "experiment", "seed",
    "git_commit", "git_dirty", "metric_name", "metric_value",
    "checkpoint_path", "log_path", "notes",
]


def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"]
        ).decode().strip()
    except Exception:
        return "unknown"


def is_git_dirty():
    try:
        status = subprocess.check_output(["git", "status", "--porcelain"]).decode()
        return bool(status.strip())
    except Exception:
        return "unknown"


def log_run(phase, experiment, seed, metric_name, metric_value,
            checkpoint_path, log_path, notes=""):
    os.makedirs(os.path.dirname(REGISTRY_PATH), exist_ok=True)
    is_new = not os.path.exists(REGISTRY_PATH)

    with open(REGISTRY_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow({
            "run_id": f"{phase}_{experiment}_{seed}_{datetime.datetime.now():%Y%m%d%H%M%S}",
            "timestamp": datetime.datetime.now().isoformat(),
            "phase": phase,
            "experiment": experiment,
            "seed": seed,
            "git_commit": get_git_commit(),
            "git_dirty": is_git_dirty(),
            "metric_name": metric_name,
            "metric_value": metric_value,
            "checkpoint_path": checkpoint_path,
            "log_path": log_path,
            "notes": notes,
        })
