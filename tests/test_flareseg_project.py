from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from framework.config import load_config
from framework.engine import build_dataloader
from projects.flare_seg.dataset import FlareSegSyntheticDataset
from projects.flare_seg.losses import DiceBCELoss


class FlareSegProjectTest(unittest.TestCase):
    def test_synthetic_dataset_returns_training_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            flickr = root / "flickr"
            flare = root / "flare"
            flickr.mkdir()
            flare.mkdir()

            Image.new("RGB", (160, 120), (80, 120, 160)).save(flickr / "base.jpg")
            flare_img = Image.new("RGB", (128, 128), (0, 0, 0))
            draw = ImageDraw.Draw(flare_img)
            draw.ellipse((50, 50, 78, 78), fill=(255, 240, 180))
            draw.line((10, 64, 118, 64), fill=(120, 90, 220), width=5)
            flare_img.save(flare / "flare.png")

            dataset = FlareSegSyntheticDataset(
                flickr_path=str(flickr),
                flare7kpp_path=str(flare),
                length=1,
                output_sizes=[[96, 192]],
                deterministic=True,
            )
            item = dataset[0]

            self.assertEqual(tuple(item["image"].shape), (3, 96, 192))
            self.assertEqual(tuple(item["mask"].shape), (1, 96, 192))
            self.assertGreater(float(item["mask"].max()), 0.0)
            self.assertLess(float(item["mask"].mean()), 1.0)

    def test_same_output_size_batch_sampler_batches_matching_shapes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            flickr = root / "flickr"
            flare = root / "flare"
            flickr.mkdir()
            flare.mkdir()

            Image.new("RGB", (160, 120), (80, 120, 160)).save(flickr / "base.jpg")
            flare_img = Image.new("RGB", (128, 128), (0, 0, 0))
            draw = ImageDraw.Draw(flare_img)
            draw.ellipse((50, 50, 78, 78), fill=(255, 240, 180))
            flare_img.save(flare / "flare.png")

            loader = build_dataloader(
                {
                    "dataset": {
                        "target": "projects.flare_seg.dataset.FlareSegSyntheticDataset",
                        "params": {
                            "flickr_path": str(flickr),
                            "flare7kpp_path": str(flare),
                            "length": 8,
                            "output_sizes": [[96, 192], [192, 96]],
                            "deterministic": True,
                            "size_selection": "index_mod",
                        },
                    },
                    "dataloader": {
                        "batch_sampler": {
                            "target": "projects.flare_seg.dataset.SameOutputSizeBatchSampler",
                            "params": {
                                "batch_size": 2,
                                "shuffle": False,
                                "drop_last": True,
                            },
                        },
                        "num_workers": 0,
                    },
                }
            )

            shapes = [tuple(batch["image"].shape) for batch in loader]

            self.assertEqual(set(shapes), {(2, 3, 96, 192), (2, 3, 192, 96)})
            self.assertEqual(len(shapes), 4)

    def test_dice_bce_loss_handles_matching_logits_and_mask(self):
        loss_fn = DiceBCELoss()
        result = loss_fn(
            logits=torch.zeros(2, 1, 96, 96),
            target=torch.ones(2, 1, 96, 96),
        )

        self.assertIn("loss", result)
        self.assertIn("iou_0.5", result)
        self.assertTrue(torch.is_tensor(result["loss"]))

    def test_training_config_uses_online_synthesis_and_wandb(self):
        cfg = load_config("configs/flareseg/train_fpn_flickr_flare7kpp.yaml")

        self.assertEqual(
            cfg.data.train.dataset.target,
            "projects.flare_seg.dataset.FlareSegSyntheticDataset",
        )
        backend_types = [backend.type for backend in cfg.logging.backends]
        self.assertIn("wandb", backend_types)
        wandb_cfg = next(backend for backend in cfg.logging.backends if backend.type == "wandb")
        self.assertTrue(wandb_cfg.enabled)
        self.assertEqual(wandb_cfg.project, "flareseg")
        image_ranges = [item.value_range for item in cfg.logging.images["items"]]
        self.assertEqual(image_ranges, ["0_1", "0_1", "0_1"])
        image_keys = [item.key for item in cfg.logging.images["items"]]
        self.assertIn("pred.mask_prob", image_keys)
        self.assertEqual(cfg.data.train.dataset.params.size_selection, "index_mod")
        self.assertEqual(
            cfg.data.train.dataloader.batch_sampler.target,
            "projects.flare_seg.dataset.SameOutputSizeBatchSampler",
        )


if __name__ == "__main__":
    unittest.main()
