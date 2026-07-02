from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from torchvision.transforms import functional as TF

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from projects.flare_seg.dataset import FlareSegSyntheticDataset
from projects.flare_seg.synthesis import tensor_to_pil_u8


def save_split(
    *,
    split: str,
    output_dir: Path,
    dataset: FlareSegSyntheticDataset,
    save_flare: bool,
) -> None:
    image_dir = output_dir / split / "images"
    mask_dir = output_dir / split / "masks"
    flare_dir = output_dir / split / "flares"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    if save_flare:
        flare_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = output_dir / split / "metadata.jsonl"
    with metadata_path.open("w", encoding="utf-8") as meta:
        for idx in range(len(dataset)):
            item = dataset[idx]
            sample_id = str(item["sample_id"])
            image_name = f"{sample_id}.png"
            mask_name = f"{sample_id}.png"

            tensor_to_pil_u8(item["viz_image"]).save(image_dir / image_name)
            TF.to_pil_image(item["mask"].detach().cpu().clamp(0.0, 1.0)).save(mask_dir / mask_name)
            if save_flare:
                tensor_to_pil_u8(item["flare"]).save(flare_dir / image_name)

            record = {
                "sample_id": sample_id,
                "image": f"images/{image_name}",
                "mask": f"masks/{mask_name}",
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
            meta.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flickr-path", default="/content/drive/MyDrive/dataset/Flickr24K.zip")
    parser.add_argument("--flare7kpp-path", default="/content/drive/MyDrive/dataset/Flare7K++.zip")
    parser.add_argument("--output-dir", default="/content/FlareSeg/data/flareseg_flickr24k_flare7kpp")
    parser.add_argument("--num-train", type=int, default=24000)
    parser.add_argument("--num-val", type=int, default=512)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--save-flare", action="store_true")
    parser.add_argument("--mask-absolute-threshold", type=float, default=0.018)
    parser.add_argument("--mask-relative-threshold", type=float, default=0.035)
    parser.add_argument("--mask-dilation", type=int, default=5)
    return parser.parse_args()


def write_dataset_card(output_dir: Path, args) -> None:
    card = f"""# FlareSeg Flickr24K x Flare7K++

Synthetic binary flare segmentation dataset generated from:

- Flickr24K backgrounds: `{args.flickr_path}`
- Flare7K++ flare assets: `{args.flare7kpp_path}`

Each sample contains an RGB image with synthetic flare and a single-channel binary mask
covering the full visible flare region, including the light source.

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
        "deterministic": True,
        "mask_absolute_threshold": args.mask_absolute_threshold,
        "mask_relative_threshold": args.mask_relative_threshold,
        "mask_dilation": args.mask_dilation,
    }
    train = FlareSegSyntheticDataset(length=args.num_train, seed=args.seed, **common)
    val = FlareSegSyntheticDataset(length=args.num_val, seed=args.seed + 1_000_000, **common)

    save_split(split="train", output_dir=output_dir, dataset=train, save_flare=args.save_flare)
    save_split(split="validation", output_dir=output_dir, dataset=val, save_flare=args.save_flare)
    print(f"saved dataset to {output_dir}")


if __name__ == "__main__":
    main()
