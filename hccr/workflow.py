"""Training, reconstruction scoring and evaluation over window datasets."""
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader

from .detection import report_scores
from .storage import restore_weights, save_weights


def seed_run(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def make_loaders(splits, batch):
    return {name: DataLoader(windows, batch_size=batch, shuffle=(name == "train"))
            for name, windows in splits.items()}


def run_epoch(model, batches, shift_scale, optimizer=None):
    training = optimizer is not None
    model.train(training)
    device = next(model.parameters()).device
    losses = []
    with torch.set_grad_enabled(training):
        for values, clock, stable_target, _ in batches:
            values, clock, stable_target = (
                x.to(device=device, dtype=torch.float32) for x in (values, clock, stable_target)
            )
            if training:
                optimizer.zero_grad()
            stable, _, recovered = model(values, clock, shift_scale)
            loss = 0.5 * mse_loss(stable, stable_target) + 0.5 * mse_loss(recovered, values)
            if training:
                loss.backward()
                optimizer.step()
            losses.append(loss.item())
    if not losses or not np.isfinite(losses).all():
        raise ValueError("training or validation produced an empty or nonfinite loss")
    return float(np.mean(losses))


@torch.inference_mode()
def score_windows(model, batches):
    """Return the final-point channel-mean squared error and label for every window."""
    model.eval()
    device = next(model.parameters()).device
    scores, labels = [], []
    for values, clock, _, truth in batches:
        values, clock = (x.to(device=device, dtype=torch.float32) for x in (values, clock))
        _, _, recovered = model(values, clock)
        errors = values[:, -1].cpu().numpy() - recovered[:, -1].cpu().numpy()
        scores.append(np.mean(errors ** 2, axis=1))
        labels.append(truth[:, -1].numpy().reshape(-1))
    return np.concatenate(scores), np.concatenate(labels)


def evaluate(model, loaders, risk):
    calibration, _ = score_windows(model, loaders["init"])
    scores, labels = score_windows(model, loaders["test"])
    return report_scores(calibration, scores, labels, risk)


def fit(model, loaders, destination, *, rounds, wait, step_size, shift_scale, risk):
    """Save the best validation checkpoint and evaluate it at the original checkpoints."""
    optimizer = torch.optim.Adam(model.parameters(), lr=step_size, weight_decay=1e-4)
    best, stale = float("inf"), 0
    for epoch in range(1, rounds + 1):
        train_loss = run_epoch(model, loaders["train"], shift_scale, optimizer)
        valid_loss = run_epoch(model, loaders["valid"], shift_scale)
        print(json.dumps(dict(epoch=epoch, train_loss=train_loss, validation_loss=valid_loss)), flush=True)
        if valid_loss <= best:
            best, stale = valid_loss, 0
            save_weights(model, destination)
        else:
            stale += 1
        if stale >= wait:
            break
        # Historical training reloaded the best weights at these validation losses.
        # This affects subsequent updates and consumes randomness during evaluation.
        if 0.6 <= valid_loss <= 12:
            restore_weights(model, destination)
            print(json.dumps(dict(epoch=epoch, evaluation=evaluate(model, loaders, risk))), flush=True)
    restore_weights(model, destination)
    return evaluate(model, loaders, risk)


def write_report(metrics, destination):
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
