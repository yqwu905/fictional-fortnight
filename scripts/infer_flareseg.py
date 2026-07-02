from __future__ import annotations

import argparse
import contextlib
import io
import re
import sys
from pathlib import Path
from typing import Mapping

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
SUPPORTED_MODEL_SIZES = ((768, 1536), (1536, 768))


def parse_size(value: str) -> tuple[int, int]:
    normalized = value.lower().replace(",", "x").replace(" ", "")
    parts = normalized.split("x")
    if len(parts) != 2:
        raise ValueError(f"expected size as HxW, got: {value}")
    height, width = int(parts[0]), int(parts[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"size must be positive, got: {value}")
    return height, width


def select_model_size(image_hw: tuple[int, int], model_size: str) -> tuple[int, int]:
    mode = str(model_size).strip().lower()
    if mode in {"original", "native", "keep"}:
        return image_hw
    if mode == "auto":
        if image_hw in SUPPORTED_MODEL_SIZES:
            return image_hw
        height, width = image_hw
        return (768, 1536) if width >= height else (1536, 768)
    return parse_size(model_size)


def iter_image_paths(input_path: Path, recursive: bool = False) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_EXTS:
            raise ValueError(f"unsupported image extension: {input_path}")
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(f"input path not found: {input_path}")

    iterator = input_path.rglob("*") if recursive else input_path.iterdir()
    return sorted(
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )


def resolve_checkpoint_path(path: str | Path, component_name: str = "segmenter") -> Path:
    root = Path(path).expanduser()
    if root.is_file():
        return root
    if not root.exists():
        raise FileNotFoundError(f"checkpoint path not found: {root}")

    direct_candidates = [
        root / "models" / f"{component_name}.pt",
        root / f"{component_name}.pt",
        root / "model.pt",
    ]
    for candidate in direct_candidates:
        if candidate.is_file():
            return candidate

    last_candidate = root / "checkpoint-last" / "models" / f"{component_name}.pt"
    if last_candidate.is_file():
        return last_candidate

    checkpoint_dirs = [
        candidate
        for candidate in root.glob("checkpoint-*")
        if (candidate / "models" / f"{component_name}.pt").is_file()
    ]
    if checkpoint_dirs:
        checkpoint_dir = max(checkpoint_dirs, key=_checkpoint_sort_key)
        return checkpoint_dir / "models" / f"{component_name}.pt"

    raise FileNotFoundError(
        f"could not find {component_name}.pt under checkpoint path: {root}"
    )


_STEP_RE = re.compile(r"checkpoint-(?:step-)?(\d+)$")


def _checkpoint_sort_key(path: Path) -> tuple[int, float]:
    match = _STEP_RE.match(path.name)
    if match:
        return int(match.group(1)), path.stat().st_mtime
    return -1, path.stat().st_mtime


def _torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _looks_like_state_dict(value) -> bool:
    if not isinstance(value, Mapping) or not value:
        return False
    return all(isinstance(key, str) for key in value.keys()) and any(
        torch.is_tensor(item) for item in value.values()
    )


def extract_state_dict(payload, component_name: str = "segmenter") -> Mapping[str, torch.Tensor]:
    if _looks_like_state_dict(payload):
        return payload

    if isinstance(payload, Mapping):
        for key in (
            "state_dict",
            "model_state_dict",
            "model",
            "net",
            "network",
            component_name,
        ):
            value = payload.get(key)
            if _looks_like_state_dict(value):
                return value

        models = payload.get("models")
        if isinstance(models, Mapping):
            value = models.get(component_name)
            if _looks_like_state_dict(value):
                return value

    raise ValueError("checkpoint does not contain a recognizable state_dict")


def strip_state_dict_prefixes(
    state_dict: Mapping[str, torch.Tensor],
    component_name: str = "segmenter",
) -> dict[str, torch.Tensor]:
    cleaned = dict(state_dict)
    prefixes = ("module.", "_orig_mod.", f"{component_name}.", "model.")

    changed = True
    while changed and cleaned:
        changed = False
        for prefix in prefixes:
            if all(key.startswith(prefix) for key in cleaned):
                cleaned = {key[len(prefix) :]: value for key, value in cleaned.items()}
                changed = True
                break

    return cleaned


def resolve_device(value: str) -> torch.device:
    requested = str(value).lower()
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(value)


def build_model(
    checkpoint_path: Path,
    device: torch.device,
    *,
    num_classes: int = 1,
    strict: bool = True,
    print_model: bool = False,
):
    from models.fpn import FPNAdvance_f4

    stdout_ctx = contextlib.nullcontext() if print_model else contextlib.redirect_stdout(io.StringIO())
    with stdout_ctx:
        model = FPNAdvance_f4(num_classes=num_classes, pretrained=False)

    payload = _torch_load(checkpoint_path)
    state_dict = extract_state_dict(payload)
    state_dict = strip_state_dict_prefixes(state_dict)
    result = model.load_state_dict(state_dict, strict=strict)
    missing = getattr(result, "missing_keys", [])
    unexpected = getattr(result, "unexpected_keys", [])

    if missing:
        print(f"[checkpoint] missing keys: {missing}")
    if unexpected:
        print(f"[checkpoint] unexpected keys: {unexpected}")

    model.to(device)
    model.eval()
    return model


def autocast_context(device: torch.device, amp: str):
    amp = str(amp).lower()
    if amp in {"no", "none", "false", "off"} or device.type == "cpu":
        return contextlib.nullcontext()
    if amp in {"bf16", "bfloat16"}:
        dtype = torch.bfloat16
    elif amp in {"fp16", "float16", "half"}:
        dtype = torch.float16
    else:
        raise ValueError(f"unsupported amp mode: {amp}")
    return torch.autocast(device_type=device.type, dtype=dtype)


def tensor_to_l_image(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.detach().float().cpu().clamp(0.0, 1.0)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    return TF.to_pil_image(tensor)


def make_overlay(image: Image.Image, mask: Image.Image, opacity: float = 0.45) -> Image.Image:
    base = image.convert("RGBA")
    alpha = mask.convert("L").point(lambda value: int(value * opacity))
    color = Image.new("RGBA", base.size, (255, 48, 48, 0))
    color.putalpha(alpha)
    return Image.alpha_composite(base, color).convert("RGB")


def output_stem_for(image_path: Path, input_root: Path, output_dir: Path) -> Path:
    if input_root.is_dir():
        relative = image_path.relative_to(input_root).with_suffix("")
    else:
        relative = Path(image_path.stem)
    return output_dir / relative


@torch.inference_mode()
def infer_one(
    model,
    image_path: Path,
    output_stem: Path,
    *,
    device: torch.device,
    model_size: str,
    threshold: float,
    amp: str,
    overlay_opacity: float,
    save_prob: bool,
    save_mask: bool,
    save_overlay: bool,
) -> dict[str, Path]:
    image = Image.open(image_path).convert("RGB")
    original_hw = (image.height, image.width)
    target_hw = select_model_size(original_hw, model_size)

    resized = TF.resize(
        image,
        list(target_hw),
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    )
    x = TF.to_tensor(resized).unsqueeze(0).to(device)

    with autocast_context(device, amp):
        logits = model(x)
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    if logits.ndim == 3:
        logits = logits.unsqueeze(1)
    if logits.ndim != 4 or logits.shape[1] != 1:
        raise RuntimeError(f"expected logits shape [B,1,H,W], got {tuple(logits.shape)}")

    prob = torch.sigmoid(logits.float())
    if tuple(prob.shape[-2:]) != original_hw:
        prob = F.interpolate(
            prob,
            size=original_hw,
            mode="bilinear",
            align_corners=False,
        )
    prob = prob[0, 0].cpu().clamp(0.0, 1.0)
    mask = (prob >= float(threshold)).float()

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}

    if save_prob:
        path = output_stem.with_name(f"{output_stem.name}_prob.png")
        tensor_to_l_image(prob).save(path)
        outputs["prob"] = path

    mask_image = tensor_to_l_image(mask)
    if save_mask:
        path = output_stem.with_name(f"{output_stem.name}_mask.png")
        mask_image.save(path)
        outputs["mask"] = path

    if save_overlay:
        path = output_stem.with_name(f"{output_stem.name}_overlay.png")
        make_overlay(image, mask_image, opacity=overlay_opacity).save(path)
        outputs["overlay"] = path

    return outputs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference for the FlareSeg FPN binary segmentation model.",
    )
    parser.add_argument("--checkpoint", required=True, help="checkpoint file, checkpoint dir, or output root")
    parser.add_argument("--input", required=True, help="image file or image directory")
    parser.add_argument("--output-dir", default="outputs/flareseg_infer")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--model-size",
        default="auto",
        help="auto, original, or an explicit HxW size. auto uses 768x1536/1536x768 by orientation.",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--amp", default="no", choices=["no", "bf16", "fp16"])
    parser.add_argument("--num-classes", type=int, default=1)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overlay-opacity", type=float, default=0.45)
    parser.add_argument("--non-strict", action="store_true", help="allow missing/unexpected checkpoint keys")
    parser.add_argument("--print-model", action="store_true", help="do not suppress FPNAdvance_f4 constructor print")
    parser.add_argument("--no-prob", action="store_true")
    parser.add_argument("--no-mask", action="store_true")
    parser.add_argument("--no-overlay", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    input_root = Path(args.input).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    device = resolve_device(args.device)

    image_paths = iter_image_paths(input_root, recursive=args.recursive)
    if args.limit and args.limit > 0:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        raise RuntimeError(f"no images found in {input_root}")

    model = build_model(
        checkpoint_path,
        device,
        num_classes=args.num_classes,
        strict=not args.non_strict,
        print_model=args.print_model,
    )

    print(f"[checkpoint] loaded {checkpoint_path}")
    print(f"[runtime] device={device} images={len(image_paths)} output_dir={output_dir}")

    for image_path in image_paths:
        output_stem = output_stem_for(image_path, input_root, output_dir)
        outputs = infer_one(
            model,
            image_path,
            output_stem,
            device=device,
            model_size=args.model_size,
            threshold=args.threshold,
            amp=args.amp,
            overlay_opacity=args.overlay_opacity,
            save_prob=not args.no_prob,
            save_mask=not args.no_mask,
            save_overlay=not args.no_overlay,
        )
        saved = ", ".join(f"{name}={path}" for name, path in outputs.items())
        print(f"[infer] {image_path} -> {saved}")


if __name__ == "__main__":
    main()
