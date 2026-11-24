"""Read historical SMD weights and write self-describing HCCR checkpoints."""
from dataclasses import asdict
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch

from .network import NetworkSpec


_PREFIX = {
    "dynamic_decomposition": "separator",
    "reconstruction": "restorer",
    "history_Transformer": "memory",
    "Linear_Seasonal": "outputs.0",
    "Linear_Trend": "outputs.1",
    "Linear_Recon": "outputs.2",
    "Cluster_assigner": "router",
}
FORMAT = "hccr/1"


def read_weights(path):
    return torch.load(Path(path), map_location="cpu", weights_only=True)


def unpack(model, saved):
    if saved.get("format") == FORMAT:
        if saved["network"] != asdict(model.spec) or (saved["channels"], saved["clock_features"]) != (
            model.channels, model.clock_features
        ):
            raise ValueError("checkpoint and model configurations differ")
        state = saved["weights"]
        if "centers" in state and "centers" not in model.state_dict():
            model.centers = torch.nn.Parameter(model.centers.detach().clone())
        model.load_state_dict(state, strict=True)
        if model.router is not None:
            with torch.no_grad():
                model.centers.copy_(saved["centers"])
        return
    translated = {}
    for key, tensor in saved.items():
        block, dot, rest = key.partition(".")
        translated[_PREFIX.get(block, block) + dot + rest] = tensor
    model.load_state_dict(translated, strict=True)


def restore_weights(model, path):
    unpack(model, read_weights(path))


def saved_spec(saved):
    if saved.get("format") == FORMAT:
        return NetworkSpec(**saved["network"])
    # The archived SMD configuration uses the historical network dimensions.
    # Group count and window length can also be read directly from the heads.
    heads = [w for key, w in saved.items() if key.startswith("Linear_Recon.linears.") and key.endswith(".weight")]
    if not heads:
        raise ValueError("expected a native checkpoint or the archived grouped SMD weights")
    return NetworkSpec(groups=len(heads), length=heads[0].shape[-1])


def save_weights(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = dict(format=FORMAT, network=asdict(model.spec), channels=model.channels,
                  clock_features=model.clock_features, weights=model.state_dict())
    if model.router is not None:
        record["centers"] = model.centers.detach()
    with NamedTemporaryFile(dir=path.parent, prefix=path.name+".", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(record, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
