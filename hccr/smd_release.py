"""Evaluate the released 33-channel SMD checkpoint with its saved calibration."""
import hashlib
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from .detection import adjusted_metrics, ranking_metrics
from .network import HCCR
from .series_io import calendar_coordinates
from .storage import saved_spec, unpack
from .workflow import seed_run


@torch.inference_mode()
def endpoint_errors(model, values, clock, batch_size, device):
    length = model.spec.length
    windows = TensorDataset(*(torch.from_numpy(a).unfold(0, length, 1).transpose(1, 2)
                              for a in (values, clock)))
    scores = []
    for inputs, times in DataLoader(windows, batch_size=batch_size, shuffle=False):
        inputs = inputs.to(device)
        recovered = model(inputs, times.to(device))[-1]
        scores.append((recovered[:, -1] - inputs[:, -1]).square().mean(dim=-1).cpu().numpy())
    result = np.concatenate(scores).astype(np.float64)
    if not np.isfinite(result).all():
        raise ValueError("nonfinite SMD endpoint scores")
    return result


def evaluate_release(weights, saved, data_root, device, batch):
    config = json.loads(weights.with_suffix(".json").read_text(encoding="utf-8"))
    with weights.open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    if (digest.hexdigest() != config["checkpoint_sha256"] or
            saved["protocol_sha256"] != config["protocol_sha256"] or
            (saved["model"], saved["dataset"], saved["seed"]) != ("hccr", "SMD", 42)):
        raise ValueError("SMD checkpoint and inference configuration do not match")
    if batch is not None and batch != config["batch_size"]:
        raise ValueError(f"This SMD checkpoint uses --batch {config['batch_size']}")
    state = saved["model_state_dict"]
    spec = saved_spec(state)
    channels = config["model_kwargs"]["feature_num"]
    folder = data_root if (data_root/"SMD_test_data.npy").is_file() else data_root/"SMD"
    values = np.load(folder/"SMD_test_data.npy")
    dates = np.load(folder/"SMD_test_date.npy").reshape(-1)
    labels = np.load(folder/"SMD_test_label.npy").reshape(-1)
    if (values.ndim != 2 or values.shape[1] != channels or len(values) < spec.length or
            len(values) != len(dates) or len(values) != len(labels) or not np.isin(labels, (0, 1)).all()):
        raise ValueError("expected aligned SMD test data, timestamps and binary labels")
    scaler = config["scaler"]
    filled = pd.DataFrame(values.astype(np.float64)).ffill().fillna(0).to_numpy()
    scaled = ((filled - scaler["mean"]) / scaler["scale"]).astype(np.float32)
    clock = calendar_coordinates(dates).astype(np.float32)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    seed_run(saved["seed"])
    model = HCCR(spec, channels, clock.shape[1], device)
    unpack(model, state)
    centers = saved["unregistered_cluster_emb"]
    if centers.shape != model.centers.shape or not torch.isfinite(centers).all():
        raise ValueError("invalid SMD cluster centers")
    model.centers.copy_(centers)
    seed_run(config["seed"])
    scores = endpoint_errors(model.eval(), scaled, clock, config["batch_size"], device)
    truth = labels[spec.length-1:].astype(bool)
    threshold = config["threshold"]
    alarms = scores > threshold
    tp = int((alarms & truth).sum())
    fp = int((alarms & ~truth).sum())
    fn = int((~alarms & truth).sum())
    return dict(test_points=len(scores), threshold=threshold,
                pointwise_precision=tp/(tp+fp) if tp+fp else 0.0,
                pointwise_recall=tp/(tp+fn) if tp+fn else 0.0,
                pointwise_f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0,
                **adjusted_metrics(truth, alarms), **ranking_metrics(truth, scores))
