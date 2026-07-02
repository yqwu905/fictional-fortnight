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
