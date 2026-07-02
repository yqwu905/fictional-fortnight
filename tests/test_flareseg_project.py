from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from framework.config import load_config
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
        self.assertEqual(image_ranges, ["0_1", "0_1"])


if __name__ == "__main__":
    unittest.main()
