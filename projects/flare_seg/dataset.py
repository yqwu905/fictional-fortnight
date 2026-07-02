from __future__ import annotations

import random
from dataclasses import replace
from typing import Iterator, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset, Sampler

from .synthesis import ImageSource, synthesize_flare_sample


FLARE7KPP_FLARE_INCLUDES = (
    "Flare7Kpp/Flare7K/Scattering_Flare/Compound_Flare/",
    "Flare7Kpp/Flare7K/Reflective_Flare/",
    "Flare7Kpp/Flare-R/Compound_Flare/",
)

FLARE7KPP_NON_FLARE_EXCLUDES = (
    "Light_Source/",
    "test_data/",
    "/input/",
    "/gt/",
    "/mask/",
)


def _looks_like_flare7kpp_path(path: str) -> bool:
    return "flare7k" in str(path).lower()


class FlareSegSyntheticDataset(Dataset):
    """On-the-fly flare segmentation data from Flickr24K backgrounds and Flare7K++ flares."""

    def __init__(
        self,
        flickr_path: str = "/content/drive/MyDrive/dataset/Flickr24K.zip",
        flare7kpp_path: str = "/content/drive/MyDrive/dataset/Flare7K++.zip",
        length: Optional[int] = None,
        image_size: int | None = None,
        output_sizes: Optional[Sequence[Sequence[int]]] = None,
        seed: int = 3407,
        deterministic: bool = False,
        normalize: bool = False,
        image_mean: Sequence[float] = (0.485, 0.456, 0.406),
        image_std: Sequence[float] = (0.229, 0.224, 0.225),
        flare_include: Optional[Sequence[str]] = None,
        flare_exclude: Optional[Sequence[str]] = None,
        mask_absolute_threshold: float = 0.018,
        mask_relative_threshold: float = 0.035,
        mask_dilation: int = 5,
        size_selection: str = "random",
    ):
        self.base_source = ImageSource(flickr_path)
        if flare_include is None and _looks_like_flare7kpp_path(flare7kpp_path):
            flare_include = FLARE7KPP_FLARE_INCLUDES
        if flare_exclude is None and _looks_like_flare7kpp_path(flare7kpp_path):
            flare_exclude = FLARE7KPP_NON_FLARE_EXCLUDES

        self.flare_source = ImageSource(
            flare7kpp_path,
            include=flare_include,
            exclude=flare_exclude,
        )
        self.length = int(length or len(self.base_source))
        self.output_sizes = self._resolve_output_sizes(image_size, output_sizes)
        self.seed = int(seed)
        self.deterministic = bool(deterministic)
        self.normalize = bool(normalize)
        self.image_mean = torch.tensor(image_mean, dtype=torch.float32).view(3, 1, 1)
        self.image_std = torch.tensor(image_std, dtype=torch.float32).view(3, 1, 1)
        self.mask_absolute_threshold = float(mask_absolute_threshold)
        self.mask_relative_threshold = float(mask_relative_threshold)
        self.mask_dilation = int(mask_dilation)
        self.size_selection = str(size_selection)

        if self.length <= 0:
            raise ValueError("length must be positive")
        if self.size_selection not in {"random", "index_mod"}:
            raise ValueError(
                "size_selection must be 'random' or 'index_mod', "
                f"got: {self.size_selection!r}"
            )

    @staticmethod
    def _resolve_output_sizes(
        image_size: int | None,
        output_sizes: Optional[Sequence[Sequence[int]]],
    ) -> Tuple[Tuple[int, int], ...]:
        if output_sizes is None:
            if image_size is None:
                output_sizes = ((768, 1536), (1536, 768))
            else:
                side = int(image_size)
                output_sizes = ((side, side),)

        sizes = []
        for size in output_sizes:
            if len(size) != 2:
                raise ValueError(f"output size must be [height, width], got: {size}")
            height, width = int(size[0]), int(size[1])
            if height <= 0 or width <= 0:
                raise ValueError(f"output size must be positive, got: {size}")
            sizes.append((height, width))
        return tuple(sizes)

    def __len__(self) -> int:
        return self.length

    def _rng(self, index: int) -> random.Random:
        if self.deterministic:
            return random.Random(self.seed + int(index))
        return random.Random(self.seed + int(index) * 1_000_003 + random.randrange(1 << 30))

    @property
    def num_size_buckets(self) -> int:
        return len(self.output_sizes)

    def size_bucket_for_index(self, index: int) -> int:
        return int(index) % self.num_size_buckets

    def output_size_for_index(self, index: int, rng: random.Random) -> Tuple[int, int]:
        if self.size_selection == "index_mod":
            return self.output_sizes[self.size_bucket_for_index(index)]
        return self.output_sizes[rng.randrange(len(self.output_sizes))]

    def __getitem__(self, index: int):
        rng = self._rng(index)
        base_index = int(index) % len(self.base_source)
        flare_index = rng.randrange(len(self.flare_source))
        output_size = self.output_size_for_index(index, rng)
        base_image = self.base_source.open_rgb(base_index)
        flare_image = self.flare_source.open_rgb(flare_index)

        image, mask, flare, record = synthesize_flare_sample(
            base_image,
            flare_image,
            rng,
            output_size=output_size,
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


class SameOutputSizeBatchSampler(Sampler[List[int]]):
    """Yield batches whose indices map to the same dataset output size."""

    def __init__(
        self,
        dataset: FlareSegSyntheticDataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = True,
        seed: int = 3407,
        rank: int = 0,
        world_size: int = 1,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not hasattr(dataset, "size_bucket_for_index"):
            raise TypeError("dataset must implement size_bucket_for_index(index)")

        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed + self.epoch)
        batches: List[List[int]] = []

        for bucket_id in range(self.dataset.num_size_buckets):
            indices = [
                index
                for index in range(len(self.dataset))
                if self.dataset.size_bucket_for_index(index) == bucket_id
            ]
            if self.shuffle:
                rng.shuffle(indices)

            for start in range(0, len(indices), self.batch_size):
                batch = indices[start : start + self.batch_size]
                if len(batch) == self.batch_size or (batch and not self.drop_last):
                    batches.append(batch)

        if self.shuffle:
            rng.shuffle(batches)

        for batch_index, batch in enumerate(batches):
            if batch_index % self.world_size == self.rank:
                yield batch

    def __len__(self) -> int:
        total = 0
        for bucket_id in range(self.dataset.num_size_buckets):
            bucket_len = sum(
                1
                for index in range(len(self.dataset))
                if self.dataset.size_bucket_for_index(index) == bucket_id
            )
            if self.drop_last:
                total += bucket_len // self.batch_size
            else:
                total += (bucket_len + self.batch_size - 1) // self.batch_size
        return (total + self.world_size - 1 - self.rank) // self.world_size
