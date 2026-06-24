from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Optional
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from omegaconf import DictConfig, OmegaConf
import torch.distributed as torch_dist

from .context import TrainContext
from .instantiate import instantiate
from .components import ComponentManager
from .optim import build_optimizers, build_schedulers
from .losses import build_losses
from .phase_runner import PhaseRunner
from .distributed import init_distributed, barrier
from .utils import StepTimer


logger = logging.getLogger(__name__)


def _plain(cfg):
    return OmegaConf.to_container(cfg, resolve=True) if isinstance(cfg, DictConfig) else cfg

def _safe_ctx_get(ctx, key, default=None):
    try:
        return ctx.get(key, required=False)
    except TypeError:
        try:
            return ctx.get(key)
        except Exception:
            return default
    except Exception:
        return default


def _to_tensorboard_images(
    value,
    *,
    value_range="auto",
    max_images=4,
):
    if value is None:
        return None

    if isinstance(value, (list, tuple)):
        value = [v for v in value if torch.is_tensor(v)]
        if not value:
            return None
        value = torch.stack(value, dim=0)

    if not torch.is_tensor(value):
        return None

    x = value.detach()

    # [C, H, W] -> [1, C, H, W]
    if x.ndim == 3:
        x = x.unsqueeze(0)

    # [H, W] -> [1, 1, H, W]
    if x.ndim == 2:
        x = x.unsqueeze(0).unsqueeze(0)

    if x.ndim != 4:
        return None

    # 支持 NHWC -> NCHW
    # 常见情况: [B, H, W, C]
    if x.shape[1] not in {1, 3, 4} and x.shape[-1] in {1, 3, 4}:
        x = x.permute(0, 3, 1, 2)

    # 只显示前 max_images 张
    x = x[: int(max_images)]

    # 只显示前 3 个通道
    if x.shape[1] > 3:
        x = x[:, :3]

    x = x.float().cpu()

    value_range = str(value_range or "auto").lower()

    if value_range in {"-1_1", "minus1_1", "neg1_1"}:
        x = (x + 1.0) / 2.0
    elif value_range in {"0_1", "01"}:
        pass
    elif value_range in {"0_255", "255"}:
        x = x / 255.0
    elif value_range == "auto":
        xmin = float(x.min())
        xmax = float(x.max())

        if xmin < -0.05:
            # 大概率是 [-1, 1]
            x = (x + 1.0) / 2.0
        elif xmax > 2.0:
            # 大概率是 [0, 255]
            x = x / 255.0
    else:
        raise ValueError(f"unsupported image value_range: {value_range}")

    x = x.clamp(0.0, 1.0)
    return x


def move_to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [move_to_device(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(move_to_device(v, device) for v in obj)
    return obj

def metric_to_scalar_tensor(value, device) -> Optional[torch.Tensor]:
    if torch.is_tensor(value):
        if value.numel() == 0:
            return None
        return value.detach().float().mean().to(device)

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return torch.tensor(float(value), device=device)

    return None


def reduce_scalar_tensor(value: torch.Tensor, average: bool = True) -> torch.Tensor:
    if torch_dist.is_available() and torch_dist.is_initialized():
        value = value.clone()
        torch_dist.all_reduce(value, op=torch_dist.ReduceOp.SUM)
        if average:
            value = value / torch_dist.get_world_size()
        return value
    return value


def build_tensorboard_writer(logging_cfg, output_dir: str, is_main_process: bool):
    logging_cfg = _plain(logging_cfg) or {}
    tb_cfg = logging_cfg.get("tensorboard", {}) or {}

    enabled = bool(tb_cfg.get("enabled", tb_cfg.get("enable", False)))
    if not enabled or not is_main_process:
        return None

    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception as e:
        raise ImportError(
            "TensorBoard logging requires tensorboard. Install with: pip install tensorboard"
        ) from e

    log_dir = tb_cfg.get("log_dir")
    if log_dir is None:
        log_dir = os.path.join(output_dir, "tensorboard")

    os.makedirs(log_dir, exist_ok=True)
    return SummaryWriter(log_dir=log_dir)

def build_dataloader(data_cfg, dist_state=None):
    data_cfg = _plain(data_cfg)
    dataset = instantiate(data_cfg["dataset"])
    dl_cfg = dict(data_cfg.get("dataloader", {}) or {})

    num_workers = int(dl_cfg.get("num_workers", 0))

    if num_workers > 0:
        dl_cfg.setdefault("pin_memory", False)
        dl_cfg.setdefault("persistent_workers", False)
        dl_cfg.setdefault("prefetch_factor", 1)
        dl_cfg.setdefault("timeout", 60)
        dl_cfg.setdefault("multiprocessing_context", "spawn")

    sampler = None

    if dist_state is not None and dist_state.enabled:
        shuffle = bool(dl_cfg.pop("shuffle", True))
        drop_last = bool(dl_cfg.get("drop_last", False))

        sampler = DistributedSampler(
            dataset,
            num_replicas=dist_state.world_size,
            rank=dist_state.rank,
            shuffle=shuffle,
            drop_last=drop_last,
        )

        dl_cfg["shuffle"] = False
        dl_cfg["sampler"] = sampler

    dataloader = DataLoader(dataset, **dl_cfg)
    return dataloader


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        runtime_cfg = _plain(cfg.get("runtime", {})) or {}

        self.dist_state = init_distributed(runtime_cfg)
        self.device = self.dist_state.device
        self.mixed_precision = runtime_cfg.get("mixed_precision", "no")
        self.global_step = 0
        profiling_cfg = _plain(cfg.get("profiling", {})) or {}
        self.profiling_enabled = bool(profiling_cfg.get("enabled", False))
        self.profiling_log_every = int(profiling_cfg.get("log_every", 1))
        checkpoint_cfg = _plain(cfg.get("checkpoint", {})) or {}
        self.save_every_steps = int(checkpoint_cfg.get("save_every_steps", 0) or 0)
        self.keep_last = checkpoint_cfg.get("keep_last", None)
        self.keep_last = int(self.keep_last) if self.keep_last not in (None, 0, "0") else None
        self.save_optimizer_state = bool(checkpoint_cfg.get("save_optimizer", True))
        self.save_scheduler_state = bool(checkpoint_cfg.get("save_scheduler", True))

        self.timer = StepTimer(
            enabled=self.profiling_enabled,
            device=self.device,
            synchronize=bool(profiling_cfg.get("synchronize", True)),
            reduce=str(profiling_cfg.get("reduce", "max")),
            is_main_process=self.dist_state.is_main_process,
        )

        experiment_cfg = _plain(cfg.get("experiment", {})) or {}
        self.output_dir = experiment_cfg.get("output_dir", "outputs")

        logging_cfg = _plain(cfg.get("logging", {})) or {}
        tb_cfg = _plain(logging_cfg.get("tensorboard", {})) or {}
        tb_image_cfg = _plain(tb_cfg.get("images", {})) or {}
        
        self.tb_image_enabled = bool(tb_image_cfg.get("enabled", False))
        self.tb_image_every_n_steps = int(tb_image_cfg.get("every_n_steps", 100) or 100)
        self.tb_image_max_images = int(tb_image_cfg.get("max_images", 4) or 4)
        self.tb_image_items = list(tb_image_cfg.get("items", []) or [])
        self.log_reduce = bool(logging_cfg.get("reduce_metrics", True))
        self.tb_writer = build_tensorboard_writer(
            logging_cfg=logging_cfg,
            output_dir=self.output_dir,
            is_main_process=self.dist_state.is_main_process,
        )

        if self.dist_state.is_main_process:
            logger.info(
                f"[runtime] distributed={self.dist_state.enabled}, "
                f"strategy={self.dist_state.strategy}, "
                f"backend={self.dist_state.backend}, "
                f"rank={self.dist_state.rank}, "
                f"local_rank={self.dist_state.local_rank}, "
                f"world_size={self.dist_state.world_size}, "
                f"device={self.device}",
            )

        self.train_loader = build_dataloader(cfg.data.train, dist_state=self.dist_state)

        self.components = ComponentManager(cfg.get("components", cfg.get("models", {}))).build_all()
        self.components.to(self.device)
        self.components.set_initial_modes()

        ddp_cfg = dict((runtime_cfg.get("distributed", {}) or {}).get("ddp", {}) or {})
        self.components.wrap_ddp(self.dist_state, ddp_cfg=ddp_cfg)

        if self.dist_state.is_main_process:
            self.components.print_parameter_summary()

        self.optimizers = build_optimizers(cfg.get("optimizers", {}), self.components)
        self.schedulers = build_schedulers(cfg.get("schedulers", {}), self.optimizers)

        self.losses = build_losses(cfg.get("losses", {}))
        for loss in self.losses.values():
            loss.to(self.device)

        self.phase_runner = PhaseRunner(
            self.components,
            self.optimizers,
            self.schedulers,
            self.losses,
            device=self.device,
            mixed_precision=self.mixed_precision,
            timer=self.timer,
        )

    def train(self):
        train_cfg = self.cfg.get("train", {})
        max_steps = int(train_cfg.get("max_steps", 1000))
        max_epochs = int(train_cfg.get("max_epochs", 10**9))
        log_every = int(train_cfg.get("log_every", 10))
        output_dir = self.output_dir

        if self.dist_state.is_main_process:
            os.makedirs(output_dir, exist_ok=True)

        if self.dist_state.enabled:
            barrier()

        for epoch in range(max_epochs):
            sampler = getattr(self.train_loader, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)

            data_t0 = time.perf_counter()
            for batch in self.train_loader:
                self.timer.reset()
                data_cost = time.perf_counter() - data_t0
                self.timer.add("step/data_wait", data_cost)

                if self.global_step >= max_steps:
                    self.save_checkpoint(output_dir, tag="last")
                    if self.tb_writer is not None:
                        self.tb_writer.flush()
                        self.tb_writer.close()
                    return

                with self.timer.time("step/move_to_device"):
                    batch = move_to_device(batch, self.device)

                ctx = TrainContext()
                ctx.set("global_step", self.global_step)
                ctx.set("epoch", epoch)
                ctx.set("rank", self.dist_state.rank)
                ctx.set("local_rank", self.dist_state.local_rank)
                ctx.set("world_size", self.dist_state.world_size)
                ctx.set("batch", batch)

                all_metrics: Dict[str, Any] = {}

                with self.timer.time("step/all_phases"):
                    for phase_cfg in self.cfg.train_program.phases:
                        metrics = self.phase_runner.run(ctx, phase_cfg)
                        all_metrics.update(metrics)

                self.log_tensorboard_images(ctx)
                scalar_metrics = self.prepare_scalar_metrics(all_metrics)
                if self.global_step % log_every == 0:
                    self.log_metrics(scalar_metrics)

                    if self.dist_state.is_main_process:
                        msg = f"step={self.global_step} " + " ".join(
                            f"{k}={v:.6g}" for k, v in scalar_metrics.items()
                        )
                        logger.info(msg)
                if (
                    self.profiling_enabled
                    and self.global_step % self.profiling_log_every == 0
                    and self.dist_state.is_main_process
                ):
                    logger.info(
                        "\n[profile] step=%s\n%s\n",
                        self.global_step,
                        self.timer.format(multiline=True),
                    )

                self.global_step += 1
                if (
                    self.save_every_steps > 0
                    and self.global_step % self.save_every_steps == 0
                ):
                    self.save_checkpoint(self.output_dir, tag=f"step-{self.global_step}")
                    self.cleanup_old_checkpoints(self.output_dir)
                data_t0 = time.perf_counter()

        self.save_checkpoint(output_dir, tag="last")
        if self.tb_writer is not None:
            self.tb_writer.flush()
            self.tb_writer.close()

    def prepare_scalar_metrics(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        scalar_metrics = {}

        for key, value in metrics.items():
            scalar_tensor = metric_to_scalar_tensor(value, self.device)
            if scalar_tensor is None:
                continue

            if self.log_reduce:
                scalar_tensor = reduce_scalar_tensor(scalar_tensor, average=True)

            scalar_metrics[key] = float(scalar_tensor.detach().cpu().item())

        return scalar_metrics

    def log_metrics(self, metrics: Dict[str, float]):
        if not self.dist_state.is_main_process:
            return

        if self.tb_writer is not None:
            for key, value in metrics.items():
                self.tb_writer.add_scalar(key, value, self.global_step)

        if self.tb_writer is not None:
            self.tb_writer.flush()

    def save_checkpoint(self, output_dir: str, tag: str = "last"):
        if not self.dist_state.is_main_process:
            return

        ckpt_dir = os.path.join(output_dir, f"checkpoint-{tag}")
        os.makedirs(ckpt_dir, exist_ok=True)

        OmegaConf.save(self.cfg, os.path.join(ckpt_dir, "config.yaml"))

        torch.save(
            {
                "global_step": self.global_step,
                "world_size": self.dist_state.world_size,
            },
            os.path.join(ckpt_dir, "trainer_state.pt"),
        )

        models_dir = os.path.join(ckpt_dir, "models")
        os.makedirs(models_dir, exist_ok=True)

        for name, module in self.components.unwrapped_items():
            if not hasattr(module, "state_dict"):
                continue

            save_policy = self.components.entries[name].cfg.get("save", "full")
            if save_policy == "none":
                continue

            if save_policy == "lora_only" and hasattr(module, "save_pretrained"):
                module.save_pretrained(os.path.join(models_dir, f"{name}_lora"))
            else:
                torch.save(
                    module.state_dict(),
                    os.path.join(models_dir, f"{name}.pt"),
                )

        if self.save_optimizer_state:
            optim_dir = os.path.join(ckpt_dir, "optimizers")
            os.makedirs(optim_dir, exist_ok=True)

            for name, optimizer in self.optimizers.items():
                torch.save(
                    optimizer.state_dict(),
                    os.path.join(optim_dir, f"{name}.pt"),
                )

        if self.save_scheduler_state:
            sched_dir = os.path.join(ckpt_dir, "schedulers")
            os.makedirs(sched_dir, exist_ok=True)

            for name, scheduler in self.schedulers.items():
                if hasattr(scheduler, "state_dict"):
                    torch.save(
                        scheduler.state_dict(),
                        os.path.join(sched_dir, f"{name}.pt"),
                    )

        logger.info("[checkpoint] saved to %s", ckpt_dir)

    def cleanup_old_checkpoints(self, output_dir: str):
        if not self.dist_state.is_main_process:
            return
    
        if self.keep_last is None or self.keep_last <= 0:
            return
    
        if not os.path.isdir(output_dir):
            return
    
        ckpts = []
    
        for name in os.listdir(output_dir):
            if not name.startswith("checkpoint-step-"):
                continue
            
            step_str = name.replace("checkpoint-step-", "")
            try:
                step = int(step_str)
            except ValueError:
                continue
            
            path = os.path.join(output_dir, name)
            if os.path.isdir(path):
                ckpts.append((step, path))
    
        ckpts.sort(key=lambda x: x[0])
    
        while len(ckpts) > self.keep_last:
            step, path = ckpts.pop(0)
            import shutil
            shutil.rmtree(path, ignore_errors=True)
            logger.info("[checkpoint] removed old checkpoint: %s", path)

    def log_tensorboard_images(self, ctx):
        if self.tb_writer is None:
            return

        if not self.tb_image_enabled:
            return

        if not self.dist_state.is_main_process:
            return

        if self.global_step % self.tb_image_every_n_steps != 0:
            return

        for item in self.tb_image_items:
            item = _plain(item)

            tag = item["tag"]
            key = item["key"]
            value_range = item.get("value_range", "auto")
            max_images = int(item.get("max_images", self.tb_image_max_images) or self.tb_image_max_images)

            value = _safe_ctx_get(ctx, key, default=None)
            images = _to_tensorboard_images(
                value,
                value_range=value_range,
                max_images=max_images,
            )

            if images is None:
                continue

            self.tb_writer.add_images(
                tag,
                images,
                global_step=self.global_step,
                dataformats="NCHW",
            )
