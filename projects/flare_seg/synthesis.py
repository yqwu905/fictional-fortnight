from __future__ import annotations

import io
import json
import math
import random
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import ColorJitter
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


@dataclass(frozen=True)
class SynthesisRecord:
    sample_id: str
    base_path: str
    flare_path: str
    gamma: float
    gain: float
    flare_dc_offset: float
    mask_ratio: float
    flare_luminance_max: float


class ImageSource:
    """Read images from either a directory tree or a zip archive."""

    def __init__(
        self,
        path: str,
        *,
        include: Optional[Sequence[str]] = None,
        exclude: Optional[Sequence[str]] = None,
    ):
        self.path = Path(path).expanduser()
        self.include = tuple(include or ())
        self.exclude = tuple(exclude or ())
        self._zip: Optional[zipfile.ZipFile] = None

        if not self.path.exists():
            raise FileNotFoundError(f"image source not found: {self.path}")

        if self.path.is_file():
            if self.path.suffix.lower() != ".zip":
                raise ValueError(f"file image source must be a zip archive: {self.path}")
            with zipfile.ZipFile(self.path) as zf:
                self.members = sorted(
                    name
                    for name in zf.namelist()
                    if _is_image_name(name)
                    and self._passes_filters(name)
                    and not name.endswith("/")
                )
        else:
            self.members = sorted(
                str(p.relative_to(self.path))
                for p in self.path.rglob("*")
                if p.is_file()
                and _is_image_name(str(p))
                and self._passes_filters(str(p.relative_to(self.path)))
            )

        if not self.members:
            raise RuntimeError(f"no images found in {self.path}")

    def _passes_filters(self, name: str) -> bool:
        if self.include and not any(token in name for token in self.include):
            return False
        if self.exclude and any(token in name for token in self.exclude):
            return False
        return True

    def __len__(self) -> int:
        return len(self.members)

    def open_rgb(self, index: int) -> Image.Image:
        member = self.members[int(index) % len(self.members)]
        if self.path.is_file():
            if self._zip is None:
                self._zip = zipfile.ZipFile(self.path)
            with self._zip.open(member) as f:
                data = f.read()
            image = Image.open(io.BytesIO(data))
        else:
            image = Image.open(self.path / member)
        return _to_rgb_on_black(image)

    def member_name(self, index: int) -> str:
        return self.members[int(index) % len(self.members)]

    def close(self) -> None:
        if self._zip is not None:
            self._zip.close()
            self._zip = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_zip"] = None
        return state


def _is_image_name(name: str) -> bool:
    return Path(name).suffix.lower() in IMAGE_EXTS


def _to_rgb_on_black(image: Image.Image) -> Image.Image:
    image.load()
    if image.mode in ("RGBA", "LA"):
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (0, 0, 0, 255))
        return Image.alpha_composite(background, rgba).convert("RGB")
    return image.convert("RGB")


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    return TF.to_tensor(image)


def tensor_to_pil_u8(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.detach().float().cpu().clamp(0.0, 1.0)
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        tensor = tensor.repeat(3, 1, 1)
    return TF.to_pil_image(tensor)


def random_resized_crop_params(
    width: int,
    height: int,
    rng: random.Random,
    *,
    scale: Tuple[float, float] = (0.55, 1.0),
    ratio: Tuple[float, float] = (0.75, 1.3333333333),
) -> Tuple[int, int, int, int]:
    area = width * height
    log_ratio = (math.log(ratio[0]), math.log(ratio[1]))

    for _ in range(10):
        target_area = area * rng.uniform(*scale)
        aspect_ratio = math.exp(rng.uniform(*log_ratio))

        crop_w = int(round(math.sqrt(target_area * aspect_ratio)))
        crop_h = int(round(math.sqrt(target_area / aspect_ratio)))
        if 0 < crop_w <= width and 0 < crop_h <= height:
            left = rng.randint(0, width - crop_w)
            top = rng.randint(0, height - crop_h)
            return top, left, crop_h, crop_w

    in_ratio = width / height
    if in_ratio < ratio[0]:
        crop_w = width
        crop_h = int(round(crop_w / ratio[0]))
    elif in_ratio > ratio[1]:
        crop_h = height
        crop_w = int(round(crop_h * ratio[1]))
    else:
        crop_w = width
        crop_h = height
    top = max(0, (height - crop_h) // 2)
    left = max(0, (width - crop_w) // 2)
    return top, left, crop_h, crop_w


def random_base_tensor(
    image: Image.Image,
    rng: random.Random,
    *,
    image_size: int,
    crop_scale: Tuple[float, float] = (0.55, 1.0),
) -> torch.Tensor:
    top, left, crop_h, crop_w = random_resized_crop_params(
        image.width,
        image.height,
        rng,
        scale=crop_scale,
    )
    image = TF.resized_crop(
        image,
        top,
        left,
        crop_h,
        crop_w,
        [image_size, image_size],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    )
    if rng.random() < 0.5:
        image = TF.hflip(image)
    return pil_to_tensor(image)


def remove_flare_background(flare: torch.Tensor) -> torch.Tensor:
    eps = 1e-7
    flat = flare.flatten(1)
    rgb_max = flat.max(dim=1).values.view(3, 1, 1)
    rgb_min = flat.min(dim=1).values.view(3, 1, 1)
    return (flare - rgb_min) * rgb_max / (rgb_max - rgb_min + eps)


def transform_flare_tensor(
    flare: torch.Tensor,
    rng: random.Random,
    *,
    image_size: int,
    min_scale: float = 0.75,
    max_scale: float = 1.35,
    max_translate: float = 0.18,
    max_degrees: float = 180.0,
) -> torch.Tensor:
    flare = TF.resize(
        flare,
        [image_size, image_size],
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )
    angle = rng.uniform(-max_degrees, max_degrees)
    scale = rng.uniform(min_scale, max_scale)
    max_shift = int(round(image_size * max_translate))
    translate = (rng.randint(-max_shift, max_shift), rng.randint(-max_shift, max_shift))
    flare = TF.affine(
        flare,
        angle=angle,
        translate=translate,
        scale=scale,
        shear=[0.0, 0.0],
        interpolation=InterpolationMode.BILINEAR,
        fill=0.0,
    )
    if rng.random() < 0.5:
        flare = TF.hflip(flare)
    if rng.random() < 0.5:
        flare = TF.vflip(flare)
    return flare


def luminance(rgb: torch.Tensor) -> torch.Tensor:
    return 0.2126 * rgb[0:1] + 0.7152 * rgb[1:2] + 0.0722 * rgb[2:3]


def make_flare_mask(
    flare_signal: torch.Tensor,
    *,
    absolute_threshold: float = 0.018,
    relative_threshold: float = 0.035,
    dilation: int = 5,
) -> torch.Tensor:
    lum = luminance(flare_signal).clamp(0.0, 1.0)
    max_value = float(lum.max().item())
    threshold = max(float(absolute_threshold), float(relative_threshold) * max_value)
    mask = (lum > threshold).float()

    if dilation and dilation > 1:
        kernel = int(dilation)
        if kernel % 2 == 0:
            kernel += 1
        mask = F.max_pool2d(
            mask.unsqueeze(0),
            kernel_size=kernel,
            stride=1,
            padding=kernel // 2,
        ).squeeze(0)
    return mask.clamp(0.0, 1.0)


def synthesize_flare_sample(
    base_image: Image.Image,
    flare_image: Image.Image,
    rng: random.Random,
    *,
    image_size: int = 384,
    mask_absolute_threshold: float = 0.018,
    mask_relative_threshold: float = 0.035,
    mask_dilation: int = 5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, SynthesisRecord]:
    gamma = rng.uniform(1.8, 2.2)
    torch_generator = torch.Generator().manual_seed(rng.randrange(2**63 - 1))
    base = random_base_tensor(base_image, rng, image_size=image_size)
    base_linear = TF.adjust_gamma(base, gamma, gain=1.0)

    sigma = 0.01 * (rng.gauss(0.0, 1.0) ** 2)
    if sigma > 0:
        base_linear = base_linear + torch.randn(
            base_linear.shape,
            generator=torch_generator,
            dtype=base_linear.dtype,
        ) * sigma
    gain = rng.uniform(0.5, 1.2)
    base_linear = (gain * base_linear).clamp(0.0, 1.0)

    flare = pil_to_tensor(flare_image)
    flare = TF.adjust_gamma(flare, gamma, gain=1.0)
    flare = remove_flare_background(flare).clamp(0.0, 1.0)
    flare = transform_flare_tensor(flare, rng, image_size=image_size)

    brightness = rng.uniform(0.8, 3.0)
    hue = rng.uniform(-0.02, 0.02)
    flare = ColorJitter(brightness=(brightness, brightness), hue=(hue, hue))(flare)
    blur_sigma = rng.uniform(0.1, 3.0)
    flare = TF.gaussian_blur(flare, kernel_size=[21, 21], sigma=[blur_sigma, blur_sigma])
    flare_signal = flare.clamp(0.0, 1.0)
    flare_dc_offset = rng.uniform(-0.02, 0.02)
    flare_linear = (flare_signal + flare_dc_offset).clamp(0.0, 1.0)

    merged_linear = (base_linear + flare_linear).clamp(0.0, 1.0)
    merged = TF.adjust_gamma(merged_linear.clamp(1e-7, 1.0), 1.0 / gamma, gain=1.0)
    flare_srgb = TF.adjust_gamma(flare_linear.clamp(1e-7, 1.0), 1.0 / gamma, gain=1.0)
    mask_signal = TF.adjust_gamma(flare_signal.clamp(1e-7, 1.0), 1.0 / gamma, gain=1.0)
    mask = make_flare_mask(
        mask_signal,
        absolute_threshold=mask_absolute_threshold,
        relative_threshold=mask_relative_threshold,
        dilation=mask_dilation,
    )

    record = SynthesisRecord(
        sample_id="",
        base_path="",
        flare_path="",
        gamma=float(gamma),
        gain=float(gain),
        flare_dc_offset=float(flare_dc_offset),
        mask_ratio=float(mask.mean().item()),
        flare_luminance_max=float(luminance(mask_signal).max().item()),
    )
    return merged, mask, flare_srgb, record


def save_record_jsonl(path: Path, records: Iterable[SynthesisRecord]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
