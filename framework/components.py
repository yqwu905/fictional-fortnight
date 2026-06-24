from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Dict, Iterable, Mapping, Optional
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from .instantiate import instantiate
from .registry import get_selector
from .distributed import DistState, wrap_ddp, unwrap_model


logger = logging.getLogger(__name__)


@dataclass
class ComponentEntry:
    name: str
    module: Any
    cfg: Mapping[str, Any]
    trainable_flags: Dict[str, bool]


class ComponentManager:
    def __init__(self, cfg: Mapping[str, Any]):
        self.cfg = OmegaConf.to_container(cfg, resolve=True) if isinstance(cfg, DictConfig) else dict(cfg or {})
        self.entries: Dict[str, ComponentEntry] = {}

    def build_all(self):
        for name, c in self.cfg.items():
            module = instantiate(c)

            ckpt = c.get("checkpoint")
            if ckpt:
                self.load_checkpoint(module, ckpt, name, strict=c.get("strict", True))

            module = self.apply_train_policy(module, c.get("train", {"strategy": "full"}))
            trainable_flags = self._snapshot_trainable_flags(module)

            self.entries[name] = ComponentEntry(
                name=name,
                module=module,
                cfg=c,
                trainable_flags=trainable_flags,
            )

        return self

    def _snapshot_trainable_flags(self, module) -> Dict[str, bool]:
        if not hasattr(module, "named_parameters"):
            return {}
        return {name: p.requires_grad for name, p in module.named_parameters()}

    def load_checkpoint(self, module, path: str, name: str, strict: bool = True):
        state = torch.load(path, map_location="cpu")

        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

        if not hasattr(module, "load_state_dict"):
            raise TypeError(f"component {module} has no load_state_dict")

        result = module.load_state_dict(state, strict=strict)

        missing = getattr(result, "missing_keys", None)
        unexpected = getattr(result, "unexpected_keys", None)

        if missing is None or unexpected is None:
            try:
                missing, unexpected = result
            except Exception:
                missing, unexpected = [], []

        logger.info("%s load from %s, result:", name, path)
        log_missing = logger.warning if missing else logger.info
        log_unexpected = logger.warning if unexpected else logger.info
        log_missing("Missing: %s", missing)
        log_unexpected("Unexpected: %s", unexpected)

    def apply_train_policy(self, module, train_cfg):
        if train_cfg is None:
            train_cfg = {"strategy": "full"}

        if isinstance(train_cfg, str):
            train_cfg = {"strategy": train_cfg}

        train_cfg = dict(train_cfg)
        strategy = train_cfg.get("strategy", "full")

        if not hasattr(module, "parameters"):
            return module

        if strategy == "frozen":
            module.requires_grad_(False)
            return module

        if strategy == "full":
            module.requires_grad_(True)
            return module

        if strategy == "lora":
            return self.apply_lora(module, train_cfg)

        if strategy == 'keep':
            return module

        raise ValueError(f"unknown train strategy: {strategy}")

    def apply_lora(self, module, train_cfg):
        try:
            from peft import LoraConfig, get_peft_model
        except Exception as e:
            raise ImportError("LoRA strategy requires `peft`. Install with `pip install peft`.") from e

        target_policy = train_cfg.get("target_policy", "all_linear")
        target_modules = train_cfg.get("target_modules")

        if target_modules is None:
            target_modules = get_selector(target_policy)(module)

        if not target_modules:
            raise ValueError(f"LoRA target policy {target_policy!r} selected no modules")

        module.requires_grad_(False)

        lora_cfg = LoraConfig(
            r=int(train_cfg.get("rank", train_cfg.get("r", 16))),
            lora_alpha=int(train_cfg.get("alpha", train_cfg.get("lora_alpha", 16))),
            lora_dropout=float(train_cfg.get("dropout", train_cfg.get("lora_dropout", 0.0))),
            target_modules=target_modules,
            bias=train_cfg.get("bias", "none"),
        )

        return get_peft_model(module, lora_cfg)

    def to(self, device):
        for entry in self.entries.values():
            if hasattr(entry.module, "to"):
                entry.module.to(device)
        return self

    def wrap_ddp(self, dist_state: DistState, ddp_cfg=None):
        if not dist_state.enabled:
            return self

        ddp_cfg = dict(ddp_cfg or {})

        for name, entry in self.entries.items():
            module = entry.module

            if not isinstance(module, nn.Module):
                continue

            has_trainable = any(p.requires_grad for p in module.parameters())
            if not has_trainable:
                continue

            component_ddp_cfg = dict(ddp_cfg)
            component_ddp_cfg.update(dict(entry.cfg.get("ddp", {}) or {}))

            entry.module = wrap_ddp(
                module,
                dist_state=dist_state,
                ddp_cfg=component_ddp_cfg,
            )

        return self

    def set_phase_state(self, trainable=None, frozen=None, modes=None):
        trainable = set(trainable or [])
        frozen = set(frozen or [])
        modes = dict(modes or {})

        for name, entry in self.entries.items():
            module = entry.module

            if hasattr(module, "requires_grad_"):
                if name in trainable:
                    self._restore_component_trainable_flags(name)
                elif name in frozen:
                    module.requires_grad_(False)

            mode = modes.get(name)

            if mode is None:
                if name in trainable and hasattr(module, "train"):
                    module.train()
                elif name in frozen and hasattr(module, "eval"):
                    module.eval()
            else:
                if mode == "train" and hasattr(module, "train"):
                    module.train()
                elif mode == "eval" and hasattr(module, "eval"):
                    module.eval()
                else:
                    raise ValueError(f"invalid mode for {name}: {mode}")

    def _restore_component_trainable_flags(self, name: str):
        entry = self.entries[name]
        module = unwrap_model(entry.module)

        if not hasattr(module, "named_parameters"):
            return

        for pname, p in module.named_parameters():
            p.requires_grad_(entry.trainable_flags.get(pname, False))

    def trainable_parameters(self, component_names: Optional[Iterable[str]] = None):
        names = list(component_names) if component_names is not None else list(self.entries)
        params = []

        for name in names:
            module = unwrap_model(self.entries[name].module)

            if hasattr(module, "parameters"):
                params.extend([p for p in module.parameters() if p.requires_grad])

        return params

    def all_parameters(self, component_names: Optional[Iterable[str]] = None):
        names = list(component_names) if component_names is not None else list(self.entries)
        params = []

        for name in names:
            module = unwrap_model(self.entries[name].module)

            if hasattr(module, "parameters"):
                params.extend(list(module.parameters()))

        return params

    def named_trainable_parameters(self, component_names: Optional[Iterable[str]] = None):
        names = list(component_names) if component_names is not None else list(self.entries)

        for cname in names:
            module = unwrap_model(self.entries[cname].module)

            if hasattr(module, "named_parameters"):
                for pname, p in module.named_parameters():
                    if p.requires_grad:
                        yield f"{cname}.{pname}", p

    def unwrap(self, name: str):
        return unwrap_model(self.entries[name].module)

    def unwrapped_items(self):
        return ((name, unwrap_model(entry.module)) for name, entry in self.entries.items())

    def get(self, name: str):
        return self.entries[name].module

    def __getitem__(self, name: str):
        return self.get(name)

    def items(self):
        return ((name, entry.module) for name, entry in self.entries.items())

    def set_initial_modes(self):
        for name, entry in self.entries.items():
            mode = entry.cfg.get("mode")
            module = entry.module

            if mode == "train" and hasattr(module, "train"):
                module.train()
            elif mode == "eval" and hasattr(module, "eval"):
                module.eval()

    def parameter_summary(self, component_names: Optional[Iterable[str]] = None):
        names = list(component_names) if component_names is not None else list(self.entries)
        summaries = {}

        for name in names:
            module = unwrap_model(self.entries[name].module)

            if not hasattr(module, "parameters"):
                summaries[name] = {
                    "total": 0,
                    "trainable": 0,
                    "frozen": 0,
                    "trainable_ratio": 0.0,
                }
                continue

            total = 0
            trainable = 0

            for p in module.parameters():
                n = p.numel()
                total += n
                if p.requires_grad:
                    trainable += n

            frozen = total - trainable
            ratio = trainable / total if total > 0 else 0.0

            summaries[name] = {
                "total": total,
                "trainable": trainable,
                "frozen": frozen,
                "trainable_ratio": ratio,
            }

        return summaries

    @staticmethod
    def _format_param_count(n: int) -> str:
        if n >= 1_000_000_000:
            return f"{n / 1_000_000_000:.3f}B"
        if n >= 1_000_000:
            return f"{n / 1_000_000:.3f}M"
        if n >= 1_000:
            return f"{n / 1_000:.3f}K"
        return str(n)

    def print_parameter_summary(self, component_names: Optional[Iterable[str]] = None):
        summaries = self.parameter_summary(component_names)

        logger.info("")
        logger.info("[component parameters]")

        total_all = 0
        trainable_all = 0

        for name, s in summaries.items():
            total = s["total"]
            trainable = s["trainable"]
            frozen = s["frozen"]
            ratio = s["trainable_ratio"]

            total_all += total
            trainable_all += trainable

            logger.info(
                f"  {name}: "
                f"trainable={self._format_param_count(trainable)} "
                f"({trainable:,}), "
                f"total={self._format_param_count(total)} "
                f"({total:,}), "
                f"frozen={self._format_param_count(frozen)} "
                f"({frozen:,}), "
                f"ratio={ratio * 100:.4f}%",
            )

        frozen_all = total_all - trainable_all
        ratio_all = trainable_all / total_all if total_all > 0 else 0.0

        logger.info(
            f"  TOTAL: "
            f"trainable={self._format_param_count(trainable_all)} "
            f"({trainable_all:,}), "
            f"total={self._format_param_count(total_all)} "
            f"({total_all:,}), "
            f"frozen={self._format_param_count(frozen_all)} "
            f"({frozen_all:,}), "
            f"ratio={ratio_all * 100:.4f}%\n",
        )
