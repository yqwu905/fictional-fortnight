from __future__ import annotations

import random
from dataclasses import replace
from typing import Optional, Sequence

import torch
from torch.utils.data import Dataset

from .synthesis import ImageSource, synthesize_flare_sample


class FlareSegSyntheticDataset(Dataset):
    """On-the-fly flare segmentation data from Flickr24K backgrounds and Flare7K++ flares."""

    def __init__(
        self,
        flickr_path: str = "/content/drive/MyDrive/dataset/Flickr24K.zip",
        flare7kpp_path: str = "/content/drive/MyDrive/dataset/Flare7K++.zip",
        length: Optional[int] = None,
        image_size: int = 384,
        seed: int = 3407,
        deterministic: bool = False,
        normalize: bool = False,
        image_mean: Sequence[float] = (0.485, 0.456, 0.406),
        image_std: Sequence[float] = (0.229, 0.224, 0.225),
        flare_include: Optional[Sequence[str]] = None,
        mask_absolute_threshold: float = 0.018,
        mask_relative_threshold: float = 0.035,
        mask_dilation: int = 5,
    ):
        self.base_source = ImageSource(flickr_path)
        self.flare_source = ImageSource(flare7kpp_path, include=flare_include)
        self.length = int(length or len(self.base_source))
        self.image_size = int(image_size)
        self.seed = int(seed)
        self.deterministic = bool(deterministic)
        self.normalize = bool(normalize)
        self.image_mean = torch.tensor(image_mean, dtype=torch.float32).view(3, 1, 1)
        self.image_std = torch.tensor(image_std, dtype=torch.float32).view(3, 1, 1)
        self.mask_absolute_threshold = float(mask_absolute_threshold)
        self.mask_relative_threshold = float(mask_relative_threshold)
        self.mask_dilation = int(mask_dilation)

        if self.length <= 0:
            raise ValueError("length must be positive")

    def __len__(self) -> int:
        return self.length

    def _rng(self, index: int) -> random.Random:
        if self.deterministic:
            return random.Random(self.seed + int(index))
        return random.Random(self.seed + int(index) * 1_000_003 + random.randrange(1 << 30))

    def __getitem__(self, index: int):
        rng = self._rng(index)
        base_index = int(index) % len(self.base_source)
        flare_index = rng.randrange(len(self.flare_source))
        base_image = self.base_source.open_rgb(base_index)
        flare_image = self.flare_source.open_rgb(flare_index)

        image, mask, flare, record = synthesize_flare_sample(
            base_image,
            flare_image,
            rng,
            image_size=self.image_size,
            mask_absolute_threshold=self.mask_absolute_threshold,
            mask_relative_threshold=self.mask_relative_threshold,
            mask_dilation=self.mask_dilation,
        )
        record = replace(
            record,
            sample_id=f"{int(index):08d}",
            base_path=self.base_source.member_name(base_index),
            flare_path=self.flare_source.member_name(flare_index),
        )

        model_image = image
        if self.normalize:
            model_image = (model_image - self.image_mean) / self.image_std

        return {
            "image": model_image,
            "mask": mask,
            "viz_image": image,
            "flare": flare,
            "sample_id": record.sample_id,
            "base_path": record.base_path,
            "flare_path": record.flare_path,
            "mask_ratio": torch.tensor(record.mask_ratio, dtype=torch.float32),
            "gamma": torch.tensor(record.gamma, dtype=torch.float32),
            "gain": torch.tensor(record.gain, dtype=torch.float32),
            "flare_dc_offset": torch.tensor(record.flare_dc_offset, dtype=torch.float32),
            "flare_luminance_max": torch.tensor(record.flare_luminance_max, dtype=torch.float32),
        }
