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

No dataset has been uploaded to Hugging Face yet. Generation/upload was intentionally stopped so it can be run on a faster machine.

## Implemented

- Added `models.fpn.FPNAdvance_f4` and its `repvit_m` dependency under `models/`.
- Added on-the-fly synthetic dataset code in `projects/flare_seg/`.
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
- Added scripts:
  - `scripts/visualize_flareseg_samples.py`
  - `scripts/prepare_flareseg_dataset.py`
  - `scripts/upload_flareseg_dataset.py`
- Prepared dataset script writes high-quality JPEG images by default and PNG masks.
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

## Generate Dataset On Faster Machine

Recommended full dataset command:

```bash
python scripts/prepare_flareseg_dataset.py \
  --output-dir /content/FlareSeg/data/flareseg_flickr24k_flare7kpp_1536x768_jpg \
  --num-train 24000 \
  --num-val 512 \
  --output-sizes 768x1536,1536x768 \
  --num-workers 4 \
  --progress-every 250 \
  --image-format jpg \
  --jpeg-quality 95
```

If the new machine has many CPU cores and fast storage, try `--num-workers 8` or `--num-workers 16`, but benchmark first. In the current Colab-like environment:

- `4` workers generated roughly dozens of images per minute.
- `8` workers had high startup overhead because every worker scans the zip archives.
- Full `24000 + 512` generation was estimated to take too long here.

For a quick sanity check before full generation:

```bash
python scripts/prepare_flareseg_dataset.py \
  --output-dir /tmp/flareseg-jpg-mini \
  --num-train 4 \
  --num-val 2 \
  --output-sizes 768x1536,1536x768 \
  --num-workers 1 \
  --progress-every 1 \
  --image-format jpg \
  --jpeg-quality 95
```

Expected structure:

```text
train/images/*.jpg
train/masks/*.png
train/metadata.jsonl
validation/images/*.jpg
validation/masks/*.png
validation/metadata.jsonl
README.md
```

`metadata.jsonl` includes `height`, `width`, `base_path`, `flare_path`, `mask_ratio`, `gamma`, `gain`, `flare_dc_offset`, and `flare_luminance_max`.

## Upload Dataset To Hugging Face

Suggested dataset repo id:

```text
yuanqingwu/flareseg-flickr24k-flare7kpp-1536x768-jpg
```

Upload after generation:

```bash
python scripts/upload_flareseg_dataset.py \
  --dataset-dir /content/FlareSeg/data/flareseg_flickr24k_flare7kpp_1536x768_jpg \
  --repo-id yuanqingwu/flareseg-flickr24k-flare7kpp-1536x768-jpg
```

Alternative with HF CLI:

```bash
hf upload yuanqingwu/flareseg-flickr24k-flare7kpp-1536x768-jpg \
  /content/FlareSeg/data/flareseg_flickr24k_flare7kpp_1536x768_jpg \
  --type dataset \
  --commit-message "Upload FlareSeg synthetic dataset"
```

Do not commit tokens or credentials to the repo.

## Train

After data generation/upload, train directly from online synthesis:

```bash
python -m framework.train --config configs/flareseg/train_fpn_flickr_flare7kpp.yaml
```

The current config still reads the local zip files and synthesizes on the fly. If you prefer training from the prepared JPEG/PNG dataset, add a file-backed dataset class or adapt the current dataset to read `metadata.jsonl`.

## Local Partial Artifacts From This Session

These were created locally and are ignored by git:

- `/content/FlareSeg/data/flareseg_flickr24k_flare7kpp`
- `/content/FlareSeg/data/flareseg_flickr24k_flare7kpp_1536x768`
- `/content/FlareSeg/data/flareseg_flickr24k_flare7kpp_1536x768_jpg`
- `outputs/flareseg_samples/`

They are partial/interrupted artifacts and should not be treated as complete datasets. It is safe to delete them before regenerating.

## Remaining Tasks

- Generate the full `24000` train and `512` validation prepared dataset on a faster machine.
- Visually inspect several generated `768x1536` and `1536x768` samples before full upload.
- Upload the prepared dataset to Hugging Face.
- Optionally add a file-backed dataset class for training from the uploaded JPEG/PNG dataset.
- Run real GPU training with `configs/flareseg/train_fpn_flickr_flare7kpp.yaml`.
- Add inference/evaluation script for real `1536x768` and `768x1536` images.

