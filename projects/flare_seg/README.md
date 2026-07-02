# FlareSeg

Train `models.fpn.FPNAdvance_f4` as a binary flare segmentation network.

Default data sources:

- `/content/drive/MyDrive/dataset/Flare7K++.zip`
- `/content/drive/MyDrive/dataset/Flickr24K.zip`

Preview synthesized image/mask pairs:

```bash
python scripts/visualize_flareseg_samples.py --samples 8
```

Generate an uploadable image/mask dataset:

```bash
python scripts/prepare_flareseg_dataset.py \
  --output-dir /content/FlareSeg/data/flareseg_flickr24k_flare7kpp \
  --num-train 24000 \
  --num-val 512 \
  --image-size 384
```

Train with the framework:

```bash
python -m framework.train --config configs/flareseg/train_fpn_flickr_flare7kpp.yaml
```
