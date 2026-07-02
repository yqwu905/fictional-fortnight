from __future__ import annotations

import io
import tempfile
import unittest
import zipfile
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from framework.config import load_config
from framework.engine import build_dataloader
from projects.flare_seg.dataset import FlareSegSyntheticDataset
from projects.flare_seg.losses import DiceBCELoss
from scripts.infer_flareseg import (
    extract_state_dict,
    infer_one,
    resolve_checkpoint_path,
    select_model_size,
    strip_state_dict_prefixes,
)


class FlareSegProjectTest(unittest.TestCase):
    @staticmethod
    def _png_bytes(color=(255, 255, 255)) -> bytes:
        buffer = io.BytesIO()
        Image.new("RGB", (16, 16), color).save(buffer, format="PNG")
        return buffer.getvalue()

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

    def test_flare7kpp_source_defaults_to_flare_only_members(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            flickr = root / "flickr"
            flickr.mkdir()
            Image.new("RGB", (64, 64), (80, 120, 160)).save(flickr / "base.jpg")

            flare_zip = root / "Flare7K++.zip"
            keep_members = [
                "Flare7Kpp/Flare7K/Scattering_Flare/Compound_Flare/000000.png",
                "Flare7Kpp/Flare7K/Reflective_Flare/000000.png",
                "Flare7Kpp/Flare-R/Compound_Flare/000000.png",
            ]
            drop_members = [
                "Flare7Kpp/Flare7K/Scattering_Flare/Light_Source/000000.png",
                "Flare7Kpp/Flare7K/Scattering_Flare/Core/000000.png",
                "Flare7Kpp/Flare7K/Scattering_Flare/Streak/000000.png",
                "Flare7Kpp/test_data/real/input/input_000000.png",
                "Flare7Kpp/test_data/real/gt/gt_000000.png",
                "Flare7Kpp/test_data/real/mask/mask_000000.png",
            ]
            with zipfile.ZipFile(flare_zip, "w") as zf:
                for member in keep_members + drop_members:
                    zf.writestr(member, self._png_bytes())

            dataset = FlareSegSyntheticDataset(
                flickr_path=str(flickr),
                flare7kpp_path=str(flare_zip),
                length=1,
                output_sizes=[[32, 32]],
                deterministic=True,
            )

            self.assertEqual(dataset.flare_source.members, sorted(keep_members))

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
            list(cfg.data.train.dataset.params.flare_include),
            [
                "Flare7Kpp/Flare7K/Scattering_Flare/Compound_Flare/",
                "Flare7Kpp/Flare7K/Reflective_Flare/",
                "Flare7Kpp/Flare-R/Compound_Flare/",
            ],
        )
        self.assertEqual(
            cfg.data.train.dataloader.batch_sampler.target,
            "projects.flare_seg.dataset.SameOutputSizeBatchSampler",
        )

    def test_inference_helpers_match_flareseg_checkpoint_contract(self):
        self.assertEqual(select_model_size((768, 1536), "auto"), (768, 1536))
        self.assertEqual(select_model_size((720, 1280), "auto"), (768, 1536))
        self.assertEqual(select_model_size((1280, 720), "auto"), (1536, 768))
        self.assertEqual(select_model_size((99, 101), "96x192"), (96, 192))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            step_dir = root / "checkpoint-step-20" / "models"
            last_dir = root / "checkpoint-last" / "models"
            step_dir.mkdir(parents=True)
            last_dir.mkdir(parents=True)

            torch.save({"module.weight": torch.ones(1)}, step_dir / "segmenter.pt")
            torch.save({"state_dict": {"segmenter.bias": torch.zeros(1)}}, last_dir / "segmenter.pt")

            self.assertEqual(resolve_checkpoint_path(root), last_dir / "segmenter.pt")

            payload = torch.load(last_dir / "segmenter.pt", map_location="cpu")
            state = extract_state_dict(payload)
            cleaned = strip_state_dict_prefixes(state)
            self.assertEqual(list(cleaned.keys()), ["bias"])

    def test_inference_one_writes_probability_mask_and_overlay(self):
        class OnesSegmenter(torch.nn.Module):
            def forward(self, x):
                return torch.ones((x.shape[0], 1, x.shape[2], x.shape[3]), device=x.device)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "input.png"
            output_stem = root / "out" / "input"
            Image.new("RGB", (32, 16), (80, 100, 120)).save(image_path)

            outputs = infer_one(
                OnesSegmenter(),
                image_path,
                output_stem,
                device=torch.device("cpu"),
                model_size="original",
                threshold=0.5,
                amp="no",
                overlay_opacity=0.45,
                save_prob=True,
                save_mask=True,
                save_overlay=True,
            )

            self.assertEqual(set(outputs), {"prob", "mask", "overlay"})
            for path in outputs.values():
                self.assertTrue(path.exists())
                self.assertEqual(Image.open(path).size, (32, 16))


if __name__ == "__main__":
    unittest.main()
