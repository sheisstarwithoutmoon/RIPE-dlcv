# Computational cost of RIPE with different backbones (DLCV course project, IIIT Sri City).
#
# Measures parameter counts (encoder, encoder+decoder, full model incl. 1x1 descriptor conv),
# forward time of the backbone network and peak GPU memory. These depend only on the
# architecture, so all models are randomly initialised and no training is needed.
#
# Timing covers model.net(img), i.e. encoder + decoder producing the heatmap and the coarse
# feature maps. Keypoint sampling and matching come afterwards and are identical for all
# backbones, so they are excluded.
#
# Usage:  python compare_backbones.py  [--size H W] [--runs N] [--cpu]

import argparse
import gc
import time

import torch

from ripe.models.backbones.resnet import ResNet
from ripe.models.backbones.vgg import VGG
from ripe.models.ripe import RIPE
from ripe.models.upsampler.hypercolumn_features import HyperColumnFeatures

torch.set_grad_enabled(False)

VARIANTS = {
    "VGG-19 (original)": lambda: VGG(pretrained=False),
    "ResNet-18 (ours)": lambda: ResNet(pretrained=False),
}


def count(m):
    return sum(p.numel() for p in m.parameters()) / 1e6


def benchmark(name, backbone, H, W, n_warmup, n_runs, dev):
    model = RIPE(net=backbone, upsampler=HyperColumnFeatures()).to(dev).eval()
    img = torch.rand(1, 3, H, W, device=dev)

    row = dict(
        name=name,
        enc=count(model.net.encoder),
        bb=count(model.net),  # encoder + decoder
        total=count(model),  # + descriptor projection
        hyper=sum(model.net.get_dim_layers_encoder()),
    )

    for _ in range(n_warmup):
        model.net(img)
    if dev.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    for _ in range(n_runs):
        model.net(img)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    row["ms"] = (time.perf_counter() - t0) / n_runs * 1000
    row["mem"] = torch.cuda.max_memory_allocated() / 1024**2 if dev.type == "cuda" else float("nan")

    # free this model before building the next one, so peak memory is per model
    del model, img
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, nargs=2, default=[480, 640], metavar=("H", "W"))
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    H, W = args.size
    names = list(VARIANTS)

    rows = [benchmark(n, VARIANTS[n](), H, W, args.warmup, args.runs, dev) for n in names]

    print(f"\nDevice: {torch.cuda.get_device_name() if dev.type == 'cuda' else 'CPU'} | input {H}x{W}\n")
    print(f"{'Model':<24}{'Hypercol':>9}{'Encoder M':>11}{'Enc+Dec M':>11}{'Total M':>9}{'ms/img':>9}{'Peak MB':>9}")
    for r in rows:
        print(
            f"{r['name']:<24}{r['hyper']:>9}{r['enc']:>11.2f}{r['bb']:>11.2f}"
            f"{r['total']:>9.2f}{r['ms']:>9.1f}{r['mem']:>9.0f}"
        )

    v = rows[0]
    print()
    for r in rows[1:]:
        print(
            f"{r['name']} vs VGG-19: {v['enc'] / r['enc']:.1f}x fewer encoder params, "
            f"{v['total'] / r['total']:.1f}x fewer total params, {v['ms'] / r['ms']:.1f}x faster"
        )


if __name__ == "__main__":
    main()
