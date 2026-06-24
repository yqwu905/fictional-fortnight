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


if __name__ == "__main__":
    unittest.main()
