# Mid-term submission: RIPE with a lightweight backbone

Vanya Awasthi, Sai Greeshma Venkatanagari (DLCV, IIIT Sri City)

Baseline: RIPE code
Glue Factory fork (https://github.com/JohannesK14/glue-factory). Everything was run on Google Colab (Tesla T4).
Drive Folder: https://drive.google.com/drive/folders/111UnnCkBgTxRUhtU72HPsa46R2ACLpVr?usp=sharing

## Our files

| File | What it is |
|---|---|
| `RIPE_mid.ipynb` | Cleaned Colab notebook with the outputs of our run (reproduction + cost comparison) |
| `ripe/models/backbones/resnet.py` | ResNet-18 backbone for RIPE (full-resolution stem, decoder built from encoder dims) |
| `compare_backbones.py` | Parameter / forward-time / peak-memory comparison of VGG-19 and ResNet-18 |
| `scripts/patch_gluefactory_poselib.py` | Fix for Glue Factory's PoseLib call, needed for MegaDepth-1500 |

All other files are the unmodified upstream RIPE code. `conf/backbones/resnet.yaml` already exists
upstream, but the module it points to (`ripe/models/backbones/resnet.py`) is not in the public release.

## How to run

```
python compare_backbones.py                              # VGG-19 vs ResNet-18 cost
python scripts/patch_gluefactory_poselib.py /path/to/glue-factory
python ripe/train.py backbones=resnet ...                # training uses the existing Hydra configs
```
