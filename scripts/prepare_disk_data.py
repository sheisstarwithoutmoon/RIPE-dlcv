# Download the DISK MegaDepth training data (and the IMW2020 validation scenes) in a form that
# fits Google Colab (DLCV course project).
#
# The full DISK training set is ~170 GB of tarballs (images + depth maps + calibration). RIPE
# training only reads the images, and ResizeAndPadWithHomography resizes every image to
# 560 px on the longer side anyway (conf/data/disk_megadepth.yaml). So for each scene we
#   - stream the tar and extract only */images/*,
#   - resize each image to the size RIPE would compute (longer side 560, other side
#     int(side * scale), same formula as ResizeAndPadWithHomography.compute_resize),
#     with an antialiased filter and JPEG quality 95,
#   - delete the original.
# Online, RIPE's resize then becomes the identity, so the network sees the same image size as
# with the original data; the only difference is that the downsampling happens once offline
# (PIL, Lanczos) instead of online (torchvision). This is a documented deviation.
#
# Scenes are added in a fixed random order (seeded) until --budget-gb of *download* volume is
# reached, so the subset is reproducible and identical for every model we train. The written
# megadepth/dataset.json only contains finished scenes and only tuples whose 3 images exist.
# Re-running continues where it stopped (finished scenes are listed in scenes_done.txt).
#
# MegaDepth-1500 test scenes (0015, 0022) are not part of the DISK training set, so there is no
# overlap with our test benchmark.
#
# Usage (Colab):
#   python scripts/prepare_disk_data.py --root /content/data/disk-data --budget-gb 40
#   tar -cf /content/drive/MyDrive/ripe_data/disk-data-560.tar -C /content/data disk-data   # cache

import argparse
import json
import random
import shutil
import subprocess
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image

BASE = "https://datasets.epfl.ch/disk-data"
TARGET = 560
IMW_SCENES = ["reichstag", "sacre_coeur", "st_peters_square"]


def fetch(url, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        subprocess.run(["wget", "-q", "--no-check-certificate", "-O", str(dst), url], check=True)


def resize_one(path):
    path = Path(path)
    try:
        im = Image.open(path)
        w, h = im.size
        if w > h:
            nw, nh = TARGET, int(h * TARGET / w)
        else:
            nw, nh = int(w * TARGET / h), TARGET
        if max(w, h) > TARGET:
            im.draft(im.mode, (nw, nh))  # fast JPEG DCT downscale (never below the target size)
            im = im.resize((nw, nh), Image.LANCZOS)
            im.save(path, quality=95)  # PIL keeps the mode (gray stays gray) and writes no EXIF
        return True
    except Exception as e:  # corrupt image -> drop it (its tuples are removed below)
        print("bad image", path, e)
        path.unlink(missing_ok=True)
        return False


def prepare_imw(root):
    d = root / "imw2020-val"
    fetch(f"{BASE}/imw2020-val/dataset.json", d / "dataset.json")
    for s in IMW_SCENES:
        if (d / "scenes" / s).exists():
            continue
        tgz = d / "scenes" / f"{s}.tar.gz"
        fetch(f"{BASE}/imw2020-val/scenes/{s}.tar.gz", tgz)
        # same extraction as data/download_disk_data.sh; validation images keep full resolution
        subprocess.run(["tar", "-xz", "--strip-components=3", "-f", str(tgz), "-C", str(d / "scenes")], check=True)
        tgz.unlink()
    print("IMW2020-val ready")


def scene_sizes():
    sizes = {}
    with urllib.request.urlopen(f"{BASE}/filesize.txt") as r:
        for line in r.read().decode().splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[1].startswith("megadepth/scenes/"):
                sizes[Path(parts[1]).stem] = int(parts[0])
    return sizes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/content/data/disk-data")
    ap.add_argument("--budget-gb", type=float, default=40.0, help="download volume of scenes to use (full set ~170)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--skip-imw", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    md = root / "megadepth"
    (md / "scenes").mkdir(parents=True, exist_ok=True)
    if not args.skip_imw:
        prepare_imw(root)

    fetch(f"{BASE}/megadepth/dataset.json", md / "dataset_full.json")
    full = json.loads((md / "dataset_full.json").read_text())

    sizes = scene_sizes()
    order = sorted(s for s in full if s in sizes)
    random.Random(args.seed).shuffle(order)
    chosen, total = [], 0
    for s in order:
        if total + sizes[s] > args.budget_gb * 1e9 and chosen:
            continue
        chosen.append(s)
        total += sizes[s]
    print(f"{len(chosen)}/{len(order)} scenes, {total / 1e9:.1f} GB to download")

    done_file = md / "scenes_done.txt"
    done = set(done_file.read_text().split()) if done_file.exists() else set()
    tmp = root / "_tmp"
    for i, s in enumerate(chosen):
        if s in done:
            continue
        print(f"[{i + 1}/{len(chosen)}] scene {s} ({sizes[s] / 1e9:.2f} GB)", flush=True)
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        # stream: no need to store the tarball; keep only the images
        subprocess.run(
            f"wget -q --no-check-certificate -O - {BASE}/megadepth/scenes/{s}.tar"
            f" | tar -x --strip-components=3 -C {tmp} --wildcards '*/images/*'",
            shell=True,
            check=True,
        )
        imgs = sorted((tmp / s / "images").glob("*"))
        with ProcessPoolExecutor(args.workers) as ex:
            list(ex.map(resize_one, imgs, chunksize=32))
        dst = md / "scenes" / s
        shutil.rmtree(dst, ignore_errors=True)
        shutil.move(str(tmp / s), str(dst))
        done.add(s)
        done_file.write_text("\n".join(sorted(done)))
    shutil.rmtree(tmp, ignore_errors=True)

    # filtered dataset.json: finished scenes only, tuples whose images all exist
    out, n_tuples = {}, 0
    for s in sorted(done):
        sc = dict(full[s])
        img_dir = md / sc["image_path"]
        exists = [(img_dir / n).exists() for n in sc["images"]]
        sc["tuples"] = [t for t in sc["tuples"] if all(exists[j] for j in t)]
        if len(sc["tuples"]) > 0:
            out[s] = sc
            n_tuples += len(sc["tuples"])
    (md / "dataset.json").write_text(json.dumps(out))
    (md / "scenes_used.txt").write_text("\n".join(out))
    print(f"dataset.json: {len(out)} scenes, {n_tuples} tuples")


if __name__ == "__main__":
    main()
