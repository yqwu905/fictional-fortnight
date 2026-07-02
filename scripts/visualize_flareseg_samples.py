from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from projects.flare_seg.dataset import FlareSegSyntheticDataset
from projects.flare_seg.synthesis import tensor_to_pil_u8


def make_overlay(image: Image.Image, mask: Image.Image) -> Image.Image:
    image = image.convert("RGBA")
    mask = mask.convert("L")
    color = Image.new("RGBA", image.size, (255, 48, 48, 0))
    color.putalpha(mask.point(lambda v: int(v * 0.45)))
    return Image.alpha_composite(image, color).convert("RGB")


def add_label(image: Image.Image, label: str) -> Image.Image:
    image = image.convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 22), fill=(0, 0, 0))
    draw.text((6, 5), label, fill=(255, 255, 255))
    return image


def build_sheet(dataset: FlareSegSyntheticDataset, samples: int, tile_size: int) -> Image.Image:
    rows = []
    for i in range(samples):
        item = dataset[i]
        image = tensor_to_pil_u8(item["viz_image"]).resize((tile_size, tile_size))
        mask = tensor_to_pil_u8(item["mask"]).convert("L").resize((tile_size, tile_size))
        flare = tensor_to_pil_u8(item["flare"]).resize((tile_size, tile_size))
        overlay = make_overlay(image, mask)
        ratio = float(item["mask_ratio"].item())

        tiles = [
            add_label(image, f"input {i}"),
            add_label(overlay, f"overlay mask_ratio={ratio:.3f}"),
            add_label(mask.convert("RGB"), "mask"),
            add_label(flare, "synthetic flare"),
        ]
        row = Image.new("RGB", (tile_size * len(tiles), tile_size), (255, 255, 255))
        for j, tile in enumerate(tiles):
            row.paste(tile, (j * tile_size, 0))
        rows.append(row)

    sheet = Image.new("RGB", (rows[0].width, tile_size * len(rows)), (255, 255, 255))
    for i, row in enumerate(rows):
        sheet.paste(row, (0, i * tile_size))
    return sheet


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flickr-path", default="/content/drive/MyDrive/dataset/Flickr24K.zip")
    parser.add_argument("--flare7kpp-path", default="/content/drive/MyDrive/dataset/Flare7K++.zip")
    parser.add_argument("--output", default="outputs/flareseg_samples/preview.png")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--mask-absolute-threshold", type=float, default=0.018)
    parser.add_argument("--mask-relative-threshold", type=float, default=0.035)
    parser.add_argument("--mask-dilation", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = FlareSegSyntheticDataset(
        flickr_path=args.flickr_path,
        flare7kpp_path=args.flare7kpp_path,
        length=args.samples,
        image_size=args.image_size,
        seed=args.seed,
        deterministic=True,
        mask_absolute_threshold=args.mask_absolute_threshold,
        mask_relative_threshold=args.mask_relative_threshold,
        mask_dilation=args.mask_dilation,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet = build_sheet(dataset, args.samples, args.tile_size)
    sheet.save(output)
    print(f"saved preview to {output}")


if __name__ == "__main__":
    main()
