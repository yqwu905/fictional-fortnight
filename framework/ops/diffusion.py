from __future__ import annotations

import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from framework.registry import register_op
from framework.resolver import resolve_input


def _plain(cfg):
    return OmegaConf.to_container(cfg, resolve=True) if isinstance(cfg, DictConfig) else cfg


@register_op("sample_timestep")
class SampleTimestepOp:
    def __init__(self, cfg):
        self.cfg = dict(_plain(cfg))

    def __call__(self, ctx, components):
        ref = ctx.get(self.cfg["ref"]) if "ref" in self.cfg else ctx.get("batch.gt")
        batch_size = ref.shape[0]
        device = ref.device
        distribution = self.cfg.get("distribution", "uniform")
        eps = float(self.cfg.get("eps", 1e-5))

        if distribution == "uniform":
            t = torch.rand(batch_size, device=device)
        elif distribution == "logit_normal":
            mean = float(self.cfg.get("mean", 0.0))
            std = float(self.cfg.get("std", 1.0))
            t = torch.randn(batch_size, device=device) * std + mean
            t = torch.sigmoid(t)
        else:
            raise ValueError(f"unknown timestep distribution: {distribution}")

        t = t.clamp(eps, 1.0 - eps)
        ctx.set(self.cfg.get("output", "noise.t"), t)


@register_op("flow_matching_prepare")
class FlowMatchingPrepareOp:
    def __init__(self, cfg):
        self.cfg = dict(_plain(cfg))

    def __call__(self, ctx, components):
        x1 = ctx.get(self.cfg["x1"])
        x0_spec = self.cfg.get("x0")
        x0 = resolve_input(x0_spec, ctx) if x0_spec is not None else torch.randn_like(x1)
        t = ctx.get(self.cfg["t"])
        view_shape = [t.shape[0]] + [1] * (x1.ndim - 1)
        tv = t.view(*view_shape)
        xt = (1.0 - tv) * x0 + tv * x1
        target_v = x1 - x0
        ctx.set(self.cfg.get("xt_output", "latent.noisy"), xt)
        ctx.set(self.cfg.get("target_output", "target.v"), target_v)


@register_op("dmd_proxy_target")
class DMDProxyTargetOp:
    """
    Minimal target-trick helper:
        target = (pred - grad * scale).detach()
    Then 0.5 * mse(pred, target) gives gradient approximately grad * scale.
    The actual teacher/guidance score computation should be implemented in a task-specific op
    and written to ctx as `grad`.
    """

    def __init__(self, cfg):
        self.cfg = dict(_plain(cfg))

    def __call__(self, ctx, components):
        pred = ctx.get(self.cfg["pred"])
        grad = ctx.get(self.cfg["grad"])
        scale = float(self.cfg.get("scale", 1.0))
        target = (pred - grad * scale).detach()
        ctx.set(self.cfg.get("output", "dmd.target"), target)
