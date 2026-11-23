# Load a RIPE model trained with ripe/train.py, for any backbone.
#
# ripe/train.py saves a plain state_dict (model_<steps>_final.pth) and, next to it, a
# backbone.yaml with the Hydra backbone config. This rebuilds exactly that architecture,
# with the same RIPE wrapper as the official vgg_hyper() (HyperColumnFeatures, 256-d descriptors),
# so the evaluation protocol is identical for the official weights and our checkpoints.
#
# Usage:
#   from ripe.model_zoo.checkpoint import load_ripe
#   model = load_ripe("runs/resnet18/model_5000_final.pth")          # backbone.yaml next to it
#   model = load_ripe("ripe_weights.pth", backbone="vgg")            # official weights

import importlib
from pathlib import Path

import torch
import yaml

from ripe.models.ripe import RIPE
from ripe.models.upsampler.hypercolumn_features import HyperColumnFeatures

BACKBONE_PRESETS = {
    "vgg": {"_target_": "ripe.models.backbones.vgg.VGG", "mode": "dect", "use_instance_norm": False},
    "resnet": {"_target_": "ripe.models.backbones.resnet.ResNet", "mode": "dect", "use_instance_norm": False},
}


def _find_backbone_cfg(model_path: Path):
    # checkpoint either in the run dir itself or in <run dir>/checkpoints/
    for d in (model_path.parent, model_path.parent.parent):
        if (d / "backbone.yaml").exists():
            return d / "backbone.yaml"
    return None


def build_backbone(cfg: dict):
    cfg = dict(cfg)
    module, cls = cfg.pop("_target_").rsplit(".", 1)
    cfg["pretrained"] = False  # weights come from the checkpoint; avoids downloading ImageNet weights
    if "stem_channels" in cfg:
        cfg["stem_channels"] = tuple(cfg["stem_channels"])
    return getattr(importlib.import_module(module), cls)(**cfg)


def load_ripe(model_path, backbone=None, map_location="cpu"):
    """backbone: None (read backbone.yaml next to the checkpoint), 'vgg', 'resnet', or a dict."""
    model_path = Path(model_path)
    if backbone is None:
        cfg_path = _find_backbone_cfg(model_path)
        if cfg_path is None:
            raise FileNotFoundError(f"No backbone.yaml next to {model_path}; pass backbone='vgg'/'resnet'.")
        backbone = yaml.safe_load(cfg_path.read_text())
    elif isinstance(backbone, str):
        backbone = BACKBONE_PRESETS[backbone]

    model = RIPE(net=build_backbone(backbone), upsampler=HyperColumnFeatures())

    state = torch.load(model_path, map_location=map_location, weights_only=False)
    if "net" in state and isinstance(state["net"], dict):  # a training checkpoint (checkpoints/last.pth)
        state = state["net"]
    # strip wrappers that Fabric / DDP may add
    for prefix in ("_forward_module.", "module.", "_orig_mod."):
        if all(k.startswith(prefix) for k in state):
            state = {k[len(prefix) :]: v for k, v in state.items()}
    model.load_state_dict(state)
    return model.eval()
