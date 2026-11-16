# Final phase: training, evaluation and ablations

Vanya Awasthi, Sai Greeshma Venkatanagari (DLCV, IIIT Sri City)

This covers the remaining work from Section 8 of the mid-term report (`group16-mtr.pdf`).
Everything runs in `RIPE_final.ipynb` on Google Colab, and all outputs are kept on Drive (`MyDrive/ripe_final`).

## What is compared

| ID | Model | Trained by us | Purpose |
|---|---|---|---|
| `official_vgg19_80k` | VGG-19, official weights (80k steps) | no | Upper reference only; it was trained on ~13x more pairs at the default budget |
| `vgg19` | VGG-19, budget B | yes | Fair baseline |
| `resnet18` | ResNet-18 + full-res stem (ours), budget B | yes | Main proposal. **`vgg19` vs `resnet18` is the core result** |
| `resnet18_layer4` | + layer4, 1024-d hypercolumn | yes | Ablation: deep context |
| `resnet18_narrowstem` | stem 16/32 ch, 480-d hypercolumn | yes | Ablation: memory and stem width |
| `resnet18_scratch` | no ImageNet weights | yes | Ablation: reliance on ImageNet init |
| `*_s1` | second seed of `vgg19` and `resnet18` | yes | Run-to-run variance of REINFORCE |

Metrics (the same for every row):
- HPatches homography AUC @1/3/5 px
- MegaDepth-1500 pose AUC @5/10/20°
- params, forward ms and peak memory (`compare_backbones.py`)
- training s/step
- IMW2020-val AUC during training (learning curves)

All evaluation uses the Glue Factory `ripe+NN` protocol from the reproduction.

Fairness: same data subset, pair budget, seed, precision and hyperparameters (`conf/train.yaml`). Only `backbones=` differs.

## Files added or changed in this phase

| File | Change |
|---|---|
| `ripe/train.py` | checkpoint and resume every `ckpt_interval` steps; CSV logs (`train_metrics.csv`, `val_metrics.csv`); `RIPE_SEED` env var; final validation; `backbone.yaml` sidecar; `val_interval=0` disables validation |
| `conf/train.yaml` | new keys `ckpt_interval`, `resume` (defaults unchanged otherwise) |
| `ripe/models/backbones/resnet.py` | `stem_channels` argument (default `(32, 64)` = mid-term model) |
| `conf/backbones/resnet_{layer4,narrowstem,scratch}.yaml` | ablation configs |
| `ripe/model_zoo/checkpoint.py` | `load_ripe()` rebuilds any trained backbone from `backbone.yaml` |
| `ripe/matcher/pose_estimator_poselib.py`, `ripe/benchmarks/imw_2020.py` | cast PoseLib inputs to float64 (same fix as for Glue Factory) |
| `scripts/prepare_disk_data.py` | downloads a reproducible DISK MegaDepth subset, keeps only images, pre-resizes them to 560 px |
| `scripts/setup_gluefactory_ckpt.py` | adds the `ripe_ckpt+NN` config to Glue Factory (= `ripe+NN` with our loader) |
| `scripts/eval_gluefactory.py` | runs HPatches and MegaDepth-1500 for one model and copies the results to Drive |
| `scripts/qualitative.py` | matches on the example pair for several models |
| `scripts/plot_results.py` | all figures and `results_table.{md,tex}` |
| `compare_backbones.py` | `--all` (ablation variants), `--json` |

## GPU plan

1. **Push this repo** to `github.com/sheisstarwithoutmoon/RIPE-dlcv`. The repo is private, so Colab gets in either with a read-only token saved as the `GH_TOKEN` Colab secret, or with a zip of the repo at `MyDrive/ripe_final/RIPE-dlcv.zip` (see notebook Section A).
2. **Section B, once:** build the data cache. Download speed sets the time, and it can be resumed.
3. **Section D:** run the timing probe, then fix `BUDGET_PAIRS`. Wall-clock time per run ≈ `NUM_STEPS × sec/step` plus 11 validations.
   - The default is 36k pairs (6000 steps × 6), which is 7.5% of the authors' budget.
   - If VGG takes about 5 s/step on a T4, one run takes about 8-9 h, i.e. 1-2 Colab sessions.
   - An L4 or A100 (Colab Pro) is faster and supports bf16, which is the authors' precision.
   - Use the same GPU type for all runs so that s/step values can be compared.
4. **Order:** E1 `vgg19` → E2 `resnet18` → evaluate both (the core deliverable) → E3 → E4 → E5 → E6 and E7 (second seeds).
   If GPU time runs out, drop items from the end of this list.
5. **Sections F–H:** evaluate, then make the figures. These can be re-run after each finished run.

## Deviations from the authors' training (report these)

- Reduced budget, with the alpha, beta and LR schedules scaled to the shorter run (this is what `num_steps` does in the config).
- A subset of the DISK scenes (listed in `scenes_used.txt`). The MegaDepth-1500 test scenes 0015 and 0022 are not in DISK, so there is no train/test overlap.
- Images are resized offline (PIL Lanczos) to the size RIPE computes online, so the network input size is unchanged.
- fp16 instead of bf16 on GPUs without bf16 (T4). The same precision is used for all runs.
