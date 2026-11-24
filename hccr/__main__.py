"""Command line entry point for training or evaluating HCCR."""
import argparse
from dataclasses import fields
import json
import os
from pathlib import Path

import torch

from .network import HCCR, NetworkSpec
from .series_io import make_splits
from .smd_release import evaluate_release
from .storage import read_weights, saved_spec, unpack
from .workflow import evaluate, fit, make_loaders, seed_run, write_report


def parse_command(argv=None):
    parser = argparse.ArgumentParser(prog="python -m hccr")
    actions = parser.add_subparsers(dest="action", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--series", default="SMD", choices=["SMD","SMAP","MSL","SWaT","PSM","WADI"])
    common.add_argument("--data-root", type=Path, required=True, help="parent of the dataset directories")
    common.add_argument("--batch", type=int, help="W96 release: 8; other runs: 40")
    common.add_argument("--cycle", type=int, default=1440, help="rolling median period")
    common.add_argument("--train-fraction", type=float, default=0.8)
    common.add_argument("--seed", type=int, default=42)
    common.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    common.add_argument("--risk", type=float, help="tail probability when estimating a new threshold")
    train = actions.add_parser("fit", parents=[common], help="train and evaluate the best weights")
    train.add_argument("--save-to", type=Path, default=Path("runs"))
    train.add_argument("--rounds", type=int, default=40)
    train.add_argument("--wait", type=int, default=3, help="early stopping patience in epochs")
    train.add_argument("--step-size", type=float, default=0.0002)
    train.add_argument("--shift-scale", type=float, default=10.0)
    train.add_argument("--repeats", type=int, default=1)
    for field in fields(NetworkSpec):
        options = {"choices": ["grouped", "separate"]} if field.name == "readout" else {}
        train.add_argument("--" + field.name.replace("_", "-"), type=type(field.default),
                           default=field.default, **options)
    score = actions.add_parser("score", parents=[common], help="evaluate saved weights")
    score.add_argument("--weights", type=Path,
                       default=Path(__file__).resolve().parents[1] / "weights" / "smd-window96.pt")
    score.add_argument("--report", type=Path, help="write the threshold and detection metrics as JSON")
    args = parser.parse_args(argv)
    if ((args.batch is not None and args.batch < 1) or args.cycle < 2 or not 0 < args.train_fraction < 1
            or (args.risk is not None and not 0 < args.risk < 1)):
        parser.error("invalid batch size, cycle, training fraction, or tail probability")
    if args.action == "fit" and (min(args.rounds, args.wait, args.repeats) < 1 or args.step_size <= 0):
        parser.error("rounds, wait, repeats and step size must be positive")
    return args


def run(argv=None):
    args = parse_command(argv)
    saved = read_weights(args.weights) if args.action == "score" else None
    released = saved is not None and "model_state_dict" in saved and "unregistered_cluster_emb" in saved
    if released:
        if args.series != "SMD" or args.risk is not None or args.seed != 42:
            raise ValueError("The W96 release uses SMD, seed 42 and its saved decision threshold")
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device.index or 0)
    if released:
        torch.set_num_threads(2)
        metrics = evaluate_release(args.weights, saved, args.data_root, device, args.batch)
        if args.report:
            write_report(metrics, args.report)
        print(json.dumps(metrics, ensure_ascii=False))
        return
    args.batch = args.batch or 40
    args.risk = args.risk or 0.01
    seed_run(args.seed)
    if args.action == "score":
        spec = saved_spec(saved)
    else:
        spec = NetworkSpec(**{f.name: getattr(args, f.name) for f in fields(NetworkSpec)})
    splits = make_splits(args.data_root, args.series, args.cycle, args.train_fraction, spec.length)
    loaders = make_loaders(splits, args.batch)
    channels = splits["train"].channel_count
    calendar_fields = splits["train"].time_width
    if args.action == "score":
        model = HCCR(spec, channels, calendar_fields, device)
        unpack(model, saved)
        metrics = evaluate(model, loaders, args.risk)
        if args.report:
            write_report(metrics, args.report)
        print(json.dumps(metrics, ensure_ascii=False))
        return
    for repeat in range(1, args.repeats + 1):
        model = HCCR(spec, channels, calendar_fields, device)
        folder = args.save_to / args.series / f"run-{repeat:02d}"
        metrics = fit(model, loaders, folder / "best.pt", rounds=args.rounds, wait=args.wait,
                      step_size=args.step_size, shift_scale=args.shift_scale, risk=args.risk)
        write_report(metrics, folder / "metrics.json")
        print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    run()
