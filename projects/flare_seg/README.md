# FlareSeg

Train `models.fpn.FPNAdvance_f4` as a binary flare segmentation network.

See `projects/flare_seg/HANDOFF.md` for the current progress, validation notes,
and training notes.

Default data sources:

- `/content/drive/MyDrive/dataset/Flare7K++.zip`
- `/content/drive/MyDrive/dataset/Flickr24K.zip`

Preview synthesized image/mask pairs:

```bash
python scripts/visualize_flareseg_samples.py --samples 8 --output-size 768x1536
```

Training does not require a prepared image/mask dataset. The framework
DataLoader instantiates `projects.flare_seg.dataset.FlareSegSyntheticDataset`,
which reads Flickr24K and Flare7K++ directly and synthesizes each sample inside
`__getitem__`.

Train with the framework:

```bash
WANDB_API_KEY=... python -m framework.train --config configs/flareseg/train_fpn_flickr_flare7kpp.yaml
```

Run inference from a framework checkpoint directory or a direct `segmenter.pt`:

```bash
python scripts/infer_flareseg.py \
  --checkpoint outputs/flareseg_fpn/checkpoint-last \
  --input /path/to/images \
  --output-dir outputs/flareseg_infer \
  --recursive
```

The script writes `*_prob.png`, `*_mask.png`, and `*_overlay.png`. By default
it resizes landscape inputs to `768x1536` and portrait inputs to `1536x768`
for the model, then resizes the probability mask back to the original image
size before saving.
