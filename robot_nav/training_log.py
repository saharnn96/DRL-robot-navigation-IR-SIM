import os
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(__file__))
LOG_PATH = os.path.join(ROOT, "TRAINING_LOG.md")

DEFAULT_FIELDS = [
    "Timestamp",
    "Script",
    "Model",
    "Model Name",
    "Device",
    "State Dim",
    "Action Dim",
    "Max Epochs",
    "Start Epoch",
    "Num Envs",
    "Parallel",
    "Batch Inference",
    "Async Training",
    "Episodes/Epoch",
    "Train Every N",
    "Training Iterations",
    "Batch Size",
    "Save Every",
    "Load Model",
]


def _ensure_header():
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w", encoding="utf-8") as f:
            f.write("# Training Log\n\n")
            f.write("| " + " | ".join(DEFAULT_FIELDS) + " |\n")
            f.write("|" + "---|" * len(DEFAULT_FIELDS) + "\n")


def _fmt(val):
    if isinstance(val, bool):
        return "Yes" if val else "No"
    if val is None:
        return ""
    return str(val)


def log_training_run(params: dict) -> bool:
    """Append a row to the training log markdown file.

    Args:
        params: mapping of known fields (keys should match the DEFAULT_FIELDS names,
            but common variants are accepted, see below).

    Returns:
        True on success, False otherwise.

    Accepted keys (case-insensitive): script, model, model_name, device, state_dim,
    action_dim, max_epochs, num_envs, parallel, batch_inference, async_training,
    episodes_per_epoch, train_every_n, training_iterations, batch_size, save_every,
    load_model.
    """
    try:
        _ensure_header()
        ts = datetime.now().isoformat(sep=" ", timespec="seconds")

        # Normalized mapping from friendly param names to the columns
        mapping = {
            "Timestamp": ts,
            "Script": params.get("script") or params.get("script_name") or "",
            "Model": params.get("model") or "",
            "Model Name": params.get("model_name") or params.get("name") or "",
            "Device": params.get("device") or "",
            "State Dim": params.get("state_dim") or "",
            "Action Dim": params.get("action_dim") or "",
            "Max Epochs": params.get("max_epochs") or "",
            "Start Epoch": params.get("start_epoch") or params.get("epoch") or "",
            "Num Envs": params.get("num_envs") or "",
            "Parallel": params.get("parallel") or params.get("enable_parallel") or False,
            "Batch Inference": params.get("batch_inference") or params.get("enable_batch_inference") or False,
            "Async Training": params.get("async_training") or params.get("enable_async_training") or False,
            "Episodes/Epoch": params.get("episodes_per_epoch") or "",
            "Train Every N": params.get("train_every_n") or "",
            "Training Iterations": params.get("training_iterations") or "",
            "Batch Size": params.get("batch_size") or "",
            "Save Every": params.get("save_every") or "",
            "Load Model": params.get("load_model") or False,
        }

        row = "| " + " | ".join(_fmt(mapping[k]) for k in DEFAULT_FIELDS) + " |\n"
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(row)
        return True
    except Exception as e:
        print("WARNING: Failed to write training log:", e)
        return False
