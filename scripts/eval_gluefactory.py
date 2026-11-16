# Run HPatches + MegaDepth-1500 (Glue Factory, ripe+NN protocol) for one model and copy the
# results to a persistent folder (DLCV course project).
#
#   official weights : python scripts/eval_gluefactory.py --tag official_vgg19_80k --official
#   our checkpoint   : python scripts/eval_gluefactory.py --tag resnet18_5k \
#                          --ckpt /content/drive/MyDrive/ripe_runs/resnet18_s32000/model_5000_final.pth
#
# Results land in <out>/<benchmark>/<tag>/{summaries.json,results.h5,conf.yaml}; plot_results.py
# reads them from there.

import argparse
import json
import shutil
import subprocess
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--ckpt", default=None)
ap.add_argument("--official", action="store_true", help="evaluate the official RIPE weights (config ripe+NN)")
ap.add_argument("--gf", default="/content/glue-factory")
ap.add_argument("--out", default="/content/drive/MyDrive/ripe_final/eval")
ap.add_argument("--benchmarks", nargs="+", default=["hpatches", "megadepth1500"])
ap.add_argument("--overwrite", action="store_true")
args = ap.parse_args()
assert args.official or args.ckpt, "pass --ckpt or --official"

gf = Path(args.gf)
for bench in args.benchmarks:
    dst = Path(args.out) / bench / args.tag
    if (dst / "summaries.json").exists() and not args.overwrite:
        print(f"[skip] {bench}/{args.tag} already evaluated")
        continue
    cmd = ["uv", "run", "python", "-m", f"gluefactory.eval.{bench}", "--tag", args.tag]
    if args.official:
        cmd += ["--conf", "ripe+NN"]
    else:
        cmd += ["--conf", "ripe_ckpt+NN", f"model.extractor.model_path={Path(args.ckpt).resolve()}"]
    if args.overwrite:
        cmd += ["--overwrite"]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=gf, check=True)

    res = gf / "outputs" / "results" / bench / args.tag
    dst.mkdir(parents=True, exist_ok=True)
    for f in ("summaries.json", "results.h5", "conf.yaml"):
        if (res / f).exists():
            shutil.copy(res / f, dst / f)
    s = json.loads((dst / "summaries.json").read_text())
    print(bench, args.tag, {k: v for k, v in s.items() if "@" in k})
