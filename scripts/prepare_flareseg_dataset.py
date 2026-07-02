from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

from torchvision.transforms import functional as TF

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from projects.flare_seg.dataset import FlareSegSyntheticDataset
from projects.flare_seg.synthesis import tensor_to_pil_u8


def parse_size(value: str) -> Tuple[int, int]:
    normalized = value.lower().replace(" ", "")
    parts = normalized.split("x")
    if len(parts) != 2:
        raise ValueError(f"expected size as HxW, got: {value}")
    height, width = int(parts[0]), int(parts[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"size must be positive, got: {value}")
    return height, width


def parse_sizes(value: str) -> Tuple[Tuple[int, int], ...]:
    return tuple(parse_size(item) for item in value.split(",") if item.strip())


def iter_worker_ranges(length: int, num_workers: int) -> Iterable[Tuple[int, int]]:
    chunk = (length + num_workers - 1) // num_workers
    for worker_id in range(num_workers):
        start = worker_id * chunk
        end = min(length, start + chunk)
        if start < end:
            yield start, end


def save_one_item(
    item,
    *,
    image_dir: Path,
    mask_dir: Path,
    flare_dir: Path,
    save_flare: bool,
    image_format: str,
    jpeg_quality: int,
) -> Dict:
    sample_id = str(item["sample_id"])
    image_format = image_format.lower()
    image_ext = "jpg" if image_format in {"jpg", "jpeg"} else "png"
    image_name = f"{sample_id}.{image_ext}"
    mask_name = f"{sample_id}.png"

    image = tensor_to_pil_u8(item["viz_image"])
    if image_ext == "jpg":
        image.save(image_dir / image_name, quality=int(jpeg_quality))
    else:
        image.save(image_dir / image_name)
    TF.to_pil_image(item["mask"].detach().cpu().clamp(0.0, 1.0)).save(mask_dir / mask_name)
    if save_flare:
        tensor_to_pil_u8(item["flare"]).save(flare_dir / image_name)

    record = {
        "sample_id": sample_id,
        "image": f"images/{image_name}",
        "mask": f"masks/{mask_name}",
        "height": int(item["viz_image"].shape[-2]),
        "width": int(item["viz_image"].shape[-1]),
        "base_path": item["base_path"],
        "flare_path": item["flare_path"],
        "mask_ratio": float(item["mask_ratio"].item()),
        "gamma": float(item["gamma"].item()),
        "gain": float(item["gain"].item()),
        "flare_dc_offset": float(item["flare_dc_offset"].item()),
        "flare_luminance_max": float(item["flare_luminance_max"].item()),
    }
    if save_flare:
        record["flare"] = f"flares/{image_name}"
    return record


def save_range(
    *,
    split: str,
    output_dir: str,
    dataset_kwargs: Dict,
    start: int,
    end: int,
    save_flare: bool,
    progress_every: int,
    image_format: str,
    jpeg_quality: int,
) -> str:
    output_dir = Path(output_dir)
    image_dir = output_dir / split / "images"
    mask_dir = output_dir / split / "masks"
    flare_dir = output_dir / split / "flares"

    dataset = FlareSegSyntheticDataset(**dataset_kwargs)
    part_path = output_dir / split / f"metadata.part-{start:08d}-{end:08d}.jsonl"
    with part_path.open("w", encoding="utf-8") as meta:
        for idx in range(start, end):
            record = save_one_item(
                dataset[idx],
                image_dir=image_dir,
                mask_dir=mask_dir,
                flare_dir=flare_dir,
                save_flare=save_flare,
                image_format=image_format,
                jpeg_quality=jpeg_quality,
            )
            meta.write(json.dumps(record, ensure_ascii=False) + "\n")
            if progress_every > 0 and (idx + 1 == end or (idx - start + 1) % progress_every == 0):
                print(
                    f"[{split}] {start:08d}-{end:08d}: {idx - start + 1}/{end - start}",
                    flush=True,
                )
    return str(part_path)


def save_split(
    *,
    split: str,
    output_dir: Path,
    dataset_kwargs: Dict,
    save_flare: bool,
    num_workers: int,
    progress_every: int,
    image_format: str,
    jpeg_quality: int,
) -> None:
    image_dir = output_dir / split / "images"
    mask_dir = output_dir / split / "masks"
    flare_dir = output_dir / split / "flares"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    if save_flare:
        flare_dir.mkdir(parents=True, exist_ok=True)

    length = int(dataset_kwargs["length"])
    metadata_path = output_dir / split / "metadata.jsonl"
    ranges = list(iter_worker_ranges(length, max(1, int(num_workers))))

    if len(ranges) == 1:
        part_paths = [
            save_range(
                split=split,
                output_dir=str(output_dir),
                dataset_kwargs=dataset_kwargs,
                start=0,
                end=length,
                save_flare=save_flare,
                progress_every=progress_every,
                image_format=image_format,
                jpeg_quality=jpeg_quality,
            )
        ]
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=len(ranges)) as pool:
            jobs = [
                pool.apply_async(
                    save_range,
                    kwds={
                        "split": split,
                        "output_dir": str(output_dir),
                        "dataset_kwargs": dataset_kwargs,
                        "start": start,
                        "end": end,
                        "save_flare": save_flare,
                        "progress_every": progress_every,
                        "image_format": image_format,
                        "jpeg_quality": jpeg_quality,
                    },
                )
                for start, end in ranges
            ]
            part_paths = [job.get() for job in jobs]

    with metadata_path.open("w", encoding="utf-8") as merged:
        for part_path in sorted(part_paths):
            part = Path(part_path)
            merged.write(part.read_text(encoding="utf-8"))
            part.unlink()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flickr-path", default="/content/drive/MyDrive/dataset/Flickr24K.zip")
    parser.add_argument("--flare7kpp-path", default="/content/drive/MyDrive/dataset/Flare7K++.zip")
    parser.add_argument(
        "--output-dir",
        default="/content/FlareSeg/data/flareseg_flickr24k_flare7kpp_1536x768_jpg",
    )
    parser.add_argument("--num-train", type=int, default=24000)
    parser.add_argument("--num-val", type=int, default=512)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--output-sizes", default="768x1536,1536x768")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--save-flare", action="store_true")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--image-format", choices=["jpg", "png"], default="jpg")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--mask-absolute-threshold", type=float, default=0.018)
    parser.add_argument("--mask-relative-threshold", type=float, default=0.035)
    parser.add_argument("--mask-dilation", type=int, default=5)
    return parser.parse_args()


def write_dataset_card(output_dir: Path, args) -> None:
    card = f"""# FlareSeg Flickr24K x Flare7K++

Synthetic binary flare segmentation dataset generated from:

- Flickr24K backgrounds: `{args.flickr_path}`
- Flare7K++ flare assets: `{args.flare7kpp_path}`

Splits:

- train: `{args.num_train}` samples
- validation: `{args.num_val}` samples

Each sample contains an RGB image with synthetic flare and a single-channel binary mask
covering the full visible flare region, including the light source. The default
generated sizes are `768x1536` and `1536x768`, matching the target deployment
image shapes.
Images are stored as `{args.image_format}` with JPEG quality `{args.jpeg_quality}` when
applicable. Masks are stored as PNG.

Generation follows the DeflareMambaV2 data construction pattern: gamma-domain base
augmentation, noise/gain perturbation, Flare7K++ background removal, color jitter,
blur, DC offset, and additive composition. The segmentation mask is derived from the
pre-offset flare residual so a positive DC offset does not turn the whole image into
foreground.
"""
    (output_dir / "README.md").write_text(card, encoding="utf-8")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_dataset_card(output_dir, args)

    common = {
        "flickr_path": args.flickr_path,
        "flare7kpp_path": args.flare7kpp_path,
        "image_size": args.image_size,
        "output_sizes": parse_sizes(args.output_sizes),
        "deterministic": True,
        "mask_absolute_threshold": args.mask_absolute_threshold,
        "mask_relative_threshold": args.mask_relative_threshold,
        "mask_dilation": args.mask_dilation,
    }
    train_kwargs = {"length": args.num_train, "seed": args.seed, **common}
    val_kwargs = {"length": args.num_val, "seed": args.seed + 1_000_000, **common}

    save_split(
        split="train",
        output_dir=output_dir,
        dataset_kwargs=train_kwargs,
        save_flare=args.save_flare,
        num_workers=args.num_workers,
        progress_every=args.progress_every,
        image_format=args.image_format,
        jpeg_quality=args.jpeg_quality,
    )
    save_split(
        split="validation",
        output_dir=output_dir,
        dataset_kwargs=val_kwargs,
        save_flare=args.save_flare,
        num_workers=args.num_workers,
        progress_every=args.progress_every,
        image_format=args.image_format,
        jpeg_quality=args.jpeg_quality,
    )
    print(f"saved dataset to {output_dir}")


if __name__ == "__main__":
    main()
