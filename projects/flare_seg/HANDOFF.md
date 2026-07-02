# FlareSeg Handoff

Last updated: 2026-07-02

## Current State

- Repository: `https://github.com/yqwu905/fictional-fortnight`
- Branch: `flareseg`
- Model target: `models.fpn.FPNAdvance_f4`
- Task: binary segmentation of the full visible flare region, including the light source.
- Training config: `configs/flareseg/train_fpn_flickr_flare7kpp.yaml`
- Data sources expected on the generation machine:
  - Flare7K++: `/content/drive/MyDrive/dataset/Flare7K++.zip`
  - Flickr24K: `/content/drive/MyDrive/dataset/Flickr24K.zip`

No prepared dataset is required for the current training path. Training reads the
source archives and synthesizes samples online in the DataLoader.

## Implemented

- Added `models.fpn.FPNAdvance_f4` and its `repvit_m` dependency under `models/`.
- Added DataLoader-time synthetic dataset code in `projects/flare_seg/`.
- Data synthesis follows the DeflareMambaV2 pattern:
  - gamma-domain base augmentation
  - noise and gain perturbation
  - Flare7K++ flare background removal
  - affine transform, color jitter, blur, DC offset
  - additive flare/base composition
- Segmentation mask is derived from the visible sRGB flare residual before DC offset, so the mask covers the visible flare/light source without turning the whole image foreground.
- Default generated/training image sizes are deployment shapes:
  - `768x1536`
  - `1536x768`
- `FPNAdvance_f4` was verified to run directly on both deployment shapes. It fails on `1024x1024`, so do not switch to 1024 without padding/cropping support.
- Added `DiceBCELoss` for binary segmentation logits.
- Added utility scripts:
  - `scripts/visualize_flareseg_samples.py`
  - `scripts/prepare_flareseg_dataset.py`
  - `scripts/upload_flareseg_dataset.py`
- `prepare_flareseg_dataset.py` and `upload_flareseg_dataset.py` are optional
  archival/debug utilities and are not part of the training path.
- Added lightweight tests in `tests/test_flareseg_project.py`.

## Validation Already Run

From `/content/FlareSeg/fictional-fortnight`:

```bash
python -m unittest tests.test_flareseg_project
```

Result: passed.

Forward shape check:

- `FPNAdvance_f4(num_classes=1)` accepts `(1, 3, 768, 1536)` and outputs `(1, 1, 768, 1536)`.
- `FPNAdvance_f4(num_classes=1)` accepts `(1, 3, 1536, 768)` and outputs `(1, 1, 1536, 768)`.
- `1024x1024` fails with a feature-size mismatch.

Framework smoke train was run on CPU for one `768x1536` sample:

```bash
python -m framework.train \
  --config configs/flareseg/train_fpn_flickr_flare7kpp.yaml \
  runtime.device=cpu \
  runtime.mixed_precision=no \
  data.train.dataset.params.length=1 \
  data.train.dataset.params.deterministic=true \
  data.train.dataset.params.output_sizes='[[768,1536]]' \
  data.train.dataloader.batch_size=1 \
  data.train.dataloader.num_workers=0 \
  data.train.dataloader.shuffle=false \
  train.max_steps=1 \
  train.log_every=1 \
  checkpoint.save_every_steps=0 \
  logging.tensorboard.enabled=false \
  logging.images.enabled=false \
  experiment.output_dir=/tmp/flareseg-smoke-768x1536
```

Result: forward, loss, backward, optimizer step, and checkpoint all completed.

Visual previews were generated and inspected locally:

- `outputs/flareseg_samples/preview_768x1536.png`
- `outputs/flareseg_samples/preview_1536x768.png`

These are ignored by git and not pushed.

## Important Training Notes

- The config uses `batch_size: 1` because the default dataset randomly alternates `768x1536` and `1536x768`. Standard PyTorch collation cannot batch tensors with different shapes.
- To use a larger batch size, either:
  - train with a single `output_sizes` value per run, or
  - add a custom collate function with padding.
- The model is the required `models.fpn.FPNAdvance_f4`; no architecture-layer changes were made.
- Input values are in `[0, 1]` by default; `normalize: false` in the training config.
- Output is a single logit channel. Use `sigmoid(logits) > 0.5` for a binary mask.

## Online Synthesis Training

Do not pre-generate a training dataset for the normal path. The config uses
`projects.flare_seg.dataset.FlareSegSyntheticDataset`, and PyTorch DataLoader
workers synthesize image/mask pairs online from:

- `/content/drive/MyDrive/dataset/Flickr24K.zip`
- `/content/drive/MyDrive/dataset/Flare7K++.zip`

The optional prepare/upload scripts remain only for offline inspection or
archival exports.

## Train

Train directly from online synthesis:

```bash
WANDB_API_KEY=<key> python -m framework.train --config configs/flareseg/train_fpn_flickr_flare7kpp.yaml
```

The current config records scalar metrics and configured images to W&B project
`flareseg`. Keep the API key in `WANDB_API_KEY`; do not commit it to the repo.

## Local Partial Artifacts From This Session

These were created locally and are ignored by git:

- `/content/FlareSeg/data/flareseg_flickr24k_flare7kpp`
- `/content/FlareSeg/data/flareseg_flickr24k_flare7kpp_1536x768`
- `/content/FlareSeg/data/flareseg_flickr24k_flare7kpp_1536x768_jpg`
- `outputs/flareseg_samples/`

They are partial/interrupted artifacts and should not be treated as complete datasets. It is safe to delete them before regenerating.

## Remaining Tasks

- Run real GPU training with `configs/flareseg/train_fpn_flickr_flare7kpp.yaml`.
- Add inference/evaluation script for real `1536x768` and `768x1536` images.
