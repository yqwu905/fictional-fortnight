from __future__ import annotations

import os
import shutil
import tempfile
import unittest

import torch
from omegaconf import OmegaConf

from framework.config import load_config
from framework.context import TrainContext
from framework.engine import Trainer, build_dataloader
from framework.loggers import LoggerCollection, _normalize_backend_configs, to_log_images
from framework.losses import build_losses
from framework.components import ComponentManager
from framework.phase_runner import PhaseRunner
from framework.optim import build_optimizers


class SmokeTrainingTest(unittest.TestCase):
    def test_dataset_batch_matches_context_contract(self):
        cfg = load_config("configs/test/tiny_regression.yaml")
        loader = build_dataloader(cfg.data.train)
        batch = next(iter(loader))

        self.assertEqual(batch["input"].shape, (4, 4))
        self.assertEqual(batch["target"].shape, (4, 2))
        self.assertEqual(batch["sample_id"].shape, (4,))

    def test_phase_runner_updates_model_and_reports_loss_metrics(self):
        torch.manual_seed(11)
        cfg = load_config("configs/test/tiny_regression.yaml")
        batch = next(iter(build_dataloader(cfg.data.train)))
        ctx = TrainContext(global_step=0, batch=batch)

        components = ComponentManager(cfg.components).build_all()
        optimizers = build_optimizers(cfg.optimizers, components)
        losses = build_losses(cfg.losses)
        runner = PhaseRunner(
            components,
            optimizers,
            schedulers={},
            losses=losses,
            device=torch.device("cpu"),
        )

        before = {
            name: param.detach().clone()
            for name, param in components.unwrap("regressor").named_parameters()
        }
        metrics = runner.run(ctx, cfg.train_program.phases[0])
        after = dict(components.unwrap("regressor").named_parameters())

        self.assertIn("mse/loss", metrics)
        self.assertIn("mse/mae", metrics)
        self.assertIn("regression/total_loss", metrics)
        self.assertTrue(ctx.has("pred.value"))
        self.assertTrue(
            any(not torch.equal(before[name], after[name]) for name in before),
            "expected at least one model parameter to change",
        )

    def test_trainer_runs_and_writes_checkpoint(self):
        cfg = load_config("configs/test/tiny_regression.yaml")
        output_dir = tempfile.mkdtemp(prefix="fictional-fortnight-test-")
        try:
            cfg = OmegaConf.merge(cfg, {"experiment": {"output_dir": output_dir}})
            trainer = Trainer(cfg)
            trainer.train()

            self.assertEqual(trainer.global_step, 3)
            self.assertTrue(os.path.exists(os.path.join(output_dir, "checkpoint-last")))
            self.assertTrue(
                os.path.exists(
                    os.path.join(output_dir, "checkpoint-last", "models", "regressor.pt")
                )
            )
            self.assertTrue(
                os.path.exists(
                    os.path.join(output_dir, "checkpoint-last", "optimizers", "main.pt")
                )
            )
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_logger_backend_configs_support_legacy_and_multi_backend_forms(self):
        legacy = _normalize_backend_configs(
            {
                "tensorboard": {
                    "enabled": True,
                    "log_dir": "/tmp/tb",
                }
            }
        )
        self.assertEqual(legacy[0]["type"], "tensorboard")
        self.assertTrue(legacy[0]["enabled"])

        multi = _normalize_backend_configs(
            {
                "backends": {
                    "tensorboard": {"enabled": True},
                    "aim": {"enabled": False},
                    "wandb": {"enabled": True, "project": "smoke"},
                }
            }
        )
        self.assertEqual([cfg["type"] for cfg in multi], ["tensorboard", "aim", "wandb"])
        self.assertEqual(multi[2]["project"], "smoke")

        shorthand = _normalize_backend_configs({"backends": ["tensorboard", "wandb"]})
        self.assertEqual([cfg["type"] for cfg in shorthand], ["tensorboard", "wandb"])

    def test_logger_collection_records_scalars_and_context_images(self):
        class RecordingLogger:
            def __init__(self):
                self.metrics = []
                self.images = []
                self.flushed = False

            def log_metrics(self, metrics, step):
                self.metrics.append((step, dict(metrics)))

            def log_images(self, tag, images, step):
                self.images.append((step, tag, tuple(images.shape)))

            def flush(self):
                self.flushed = True

            def close(self):
                self.flush()

        backend = RecordingLogger()
        collection = LoggerCollection(
            [backend],
            image_cfg={
                "enabled": True,
                "every_n_steps": 2,
                "max_images": 2,
                "items": [
                    {
                        "tag": "image/input",
                        "key": "batch.image",
                        "value_range": "-1_1",
                    }
                ],
            },
        )
        ctx = TrainContext()
        ctx.set("batch.image", torch.zeros(3, 4, 5))

        collection.log_metrics({"loss": 1.25}, step=1)
        collection.log_images_from_context(ctx, step=1)
        collection.log_images_from_context(ctx, step=2)
        collection.flush()

        self.assertEqual(backend.metrics, [(1, {"loss": 1.25})])
        self.assertEqual(backend.images, [(2, "image/input", (1, 3, 4, 5))])
        self.assertTrue(backend.flushed)

    def test_to_log_images_handles_nhwc_and_value_ranges(self):
        value = torch.full((2, 4, 5, 3), 255.0)
        images = to_log_images(value, value_range="0_255", max_images=1)
        self.assertEqual(tuple(images.shape), (1, 3, 4, 5))
        self.assertTrue(torch.allclose(images, torch.ones_like(images)))


if __name__ == "__main__":
    unittest.main()
