# Qualitative comparison on the repository's example pair (DLCV course project).
# Same protocol as Figure 2 of the mid-term report: up to 2048 keypoints, threshold 0.5,
# mutual NN matching, Kornia fundamental-matrix RANSAC at 1 px (on CPU, see mid-term notes).
#
# Usage:
#   python scripts/qualitative.py --out fig_qualitative.png \
#       --model "VGG-19 official (80k)=official" \
#       --model "VGG-19 (retrained)=/content/drive/MyDrive/ripe_final/runs/vgg19_s32000/model_5000_final.pth" \
#       --model "ResNet-18 (ours)=/content/drive/MyDrive/ripe_final/runs/resnet18_s32000/model_5000_final.pth"

import argparse
import sys
from pathlib import Path

import cv2
import kornia.feature as KF
import kornia.geometry as KG
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.io import decode_image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ripe import vgg_hyper  # noqa: E402
from ripe.model_zoo.checkpoint import load_ripe  # noqa: E402
from ripe.utils.utils import cv2_matches_from_kornia, resize_image, to_cv_kpts  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", action="append", required=True, help='"Label=path" or "Label=official"')
ap.add_argument("--img1", default="assets/all_souls_000013.jpg")
ap.add_argument("--img2", default="assets/all_souls_000055.jpg")
ap.add_argument("--out", default="fig_qualitative.png")
args = ap.parse_args()

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root = Path(__file__).resolve().parent.parent
im1 = resize_image(decode_image(str(root / args.img1)).float().to(dev) / 255.0)
im2 = resize_image(decode_image(str(root / args.img2)).float().to(dev) / 255.0)
np1 = (im1.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
np2 = (im2.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)

rows = []
for spec in args.model:
    label, path = spec.split("=", 1)
    model = vgg_hyper() if path == "official" else load_ripe(path)
    model = model.to(dev).eval()
    torch.manual_seed(0)
    k1, d1, s1 = model.detectAndCompute(im1, threshold=0.5, top_k=2048)
    k2, d2, s2 = model.detectAndCompute(im2, threshold=0.5, top_k=2048)
    dists, idxs = KF.DescriptorMatcher("mnn")(d1, d2)
    torch.manual_seed(0)
    _, mask = KG.ransac.RANSAC(model_type="fundamental", inl_th=1.0)(k1[idxs[:, 0]].cpu(), k2[idxs[:, 1]].cpu())
    n_in = int(mask.sum())
    vis = cv2.drawMatches(
        np1, to_cv_kpts(k1, s1), np2, to_cv_kpts(k2, s2), cv2_matches_from_kornia(dists, idxs), None,
        matchColor=(0, 255, 0), matchesMask=mask.int().ravel().tolist(), singlePointColor=(0, 0, 255),
    )
    txt = f"{label}: {len(k1)}/{len(k2)} kpts, {len(idxs)} matches, {n_in} inliers ({100 * n_in / max(len(idxs), 1):.1f}%)"
    print(txt)
    rows.append((vis, txt))
    del model
    torch.cuda.empty_cache()

fig, axes = plt.subplots(len(rows), 1, figsize=(14, 5.2 * len(rows)))
for ax, (vis, txt) in zip(np.atleast_1d(axes), rows):
    ax.imshow(vis)
    ax.set_title(txt, loc="left", fontsize=11)
    ax.axis("off")
fig.tight_layout()
fig.savefig(args.out, dpi=110, bbox_inches="tight")
print("saved", args.out)
