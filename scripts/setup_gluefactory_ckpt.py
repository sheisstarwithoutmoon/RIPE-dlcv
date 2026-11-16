# Make the RIPE fork of Glue Factory evaluate OUR checkpoints (any backbone) with the
# unchanged ripe+NN protocol (DLCV course project).
#
# What it does:
#   1. points glue-factory/thirdparty/ripe at this repository (a superset of the upstream RIPE
#      code: same vgg_hyper(), plus resnet.py and model_zoo/checkpoint.py), so the official
#      `ripe+NN` config keeps working and the new extractor can build ResNet models;
#   2. adds gluefactory/models/extractors/ripe_ckpt.py: identical to the fork's extractors/ripe.py
#      except that the model is built with load_ripe(model_path) instead of vgg_hyper();
#   3. adds gluefactory/configs/ripe_ckpt+NN.yaml: a copy of ripe+NN.yaml with only the
#      extractor name changed, so thresholds, top_k, resolutions and RANSAC are the same;
#   4. applies the PoseLib signature fix (scripts/patch_gluefactory_poselib.py).
#
# Usage:  python scripts/setup_gluefactory_ckpt.py /content/glue-factory /content/RIPE

import os
import shutil
import subprocess
import sys
from pathlib import Path

gf = Path(sys.argv[1] if len(sys.argv) > 1 else "/content/glue-factory").resolve()
ripe_repo = Path(sys.argv[2] if len(sys.argv) > 2 else Path(__file__).parent.parent).resolve()

# 1. thirdparty/ripe -> this repo
tp = gf / "thirdparty" / "ripe"
if tp.is_symlink() or not (tp / "ripe" / "models" / "backbones" / "resnet.py").exists():
    if tp.is_symlink() or tp.is_file():
        tp.unlink()
    elif tp.exists():
        shutil.rmtree(tp)
    tp.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(ripe_repo, tp, target_is_directory=True)
print("thirdparty/ripe ->", tp.resolve())

# 2. extractor
EXTRACTOR = '''import sys
from pathlib import Path

import torch
import torchvision.transforms as transforms

from ..base_model import BaseModel

ripe_path = Path(__file__).parent / "../../../thirdparty/ripe"
if not ripe_path.exists():
    raise RuntimeError(f"RIPE path not found: {ripe_path}")
if str(ripe_path) not in sys.path:
    sys.path.append(str(ripe_path))

from ripe.model_zoo.checkpoint import load_ripe


class RIPECkpt(BaseModel):
    # same defaults and forward as extractors/ripe.py; only model loading differs
    default_conf = {
        "name": "RIPECkpt",
        "model_path": None,  # model_<steps>_final.pth with backbone.yaml next to it
        "backbone": None,  # None -> read backbone.yaml; or "vgg" / "resnet"
        "chunk": 4,
        "dense_outputs": False,
        "threshold": 1.0,
        "top_k": 2048,
    }

    required_data_keys = ["image"]

    def _init(self, conf):
        if conf.model_path is None:
            raise ValueError("set model.extractor.model_path=<checkpoint>")
        self.normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.model = load_ripe(conf.model_path, backbone=conf.backbone)
        self.model.eval()
        self.set_initialized()

    def _forward(self, data):
        image = data["image"]
        keypoints, scores, descriptors = [], [], []
        chunk = self.conf.chunk
        for i in range(0, image.shape[0], chunk):
            if self.conf.dense_outputs:
                raise NotImplementedError("Dense outputs are not supported")
            im = image[: min(image.shape[0], i + chunk)]
            im = self.normalizer(im)
            kpt, desc, score = self.model.detectAndCompute(im, threshold=self.conf.threshold, top_k=self.conf.top_k)
            keypoints += [kpt.squeeze(0)]
            scores += [score.squeeze(0)]
            descriptors += [desc.squeeze(0)]
        return {
            "keypoints": torch.stack(keypoints, 0).to(image),
            "keypoint_scores": torch.stack(scores, 0).to(image),
            "descriptors": torch.stack(descriptors, 0).to(image),
        }

    def loss(self, pred, data):
        raise NotImplementedError
'''
(gf / "gluefactory/models/extractors/ripe_ckpt.py").write_text(EXTRACTOR)

# 3. config = ripe+NN.yaml with the extractor swapped
src = (gf / "gluefactory/configs/ripe+NN.yaml").read_text()
assert "name: extractors.ripe\n" in src, "ripe+NN.yaml layout changed"
(gf / "gluefactory/configs/ripe_ckpt+NN.yaml").write_text(
    src.replace("name: extractors.ripe\n", "name: extractors.ripe_ckpt\n")
)
print("wrote extractors/ripe_ckpt.py and configs/ripe_ckpt+NN.yaml")

# 4. PoseLib fix
subprocess.run([sys.executable, str(ripe_repo / "scripts/patch_gluefactory_poselib.py"), str(gf)], check=True)
