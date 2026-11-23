"""Build the demo page used by boundary_edit_demo.sh.

One 64x64 page: a paper background (label 240), a grey rule band
(label 96) on the left, a blue sky band (label 32) on the right, and a
one-column red seal (label 160) inside the sky band.  The edit rectangle
(x=24, y=20, 16x12) straddles the rule/sky boundary.

In the original photo the highest-contrast pixel pairs sit just right of
the seal column (columns 37/38/39), so the unconstrained best boundary
hugs the seal and creates a rule/seal contact.  The sky already touches
the seal outside the rectangle, and the rule always touches the sky after
any edit, so once the seal is locked to the palette red — which only the
paper colour is far enough from — every boundary that touches the seal
becomes infeasible and the best feasible boundary retreats to the left of
the seal column.  That is what the demo shows through the API.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

# Same palette as the colouring walkthrough.
PALETTE = ["#101010", "#f2eee2", "#a12f26", "#2f5d8a", "#d8b41e"]
MIN_DISTANCE = 200

PAPER = 240
RULE = 96
SKY = 32
SEAL = 160

WIDTH = HEIGHT = 64
# The edit rectangle used by boundary_edit_demo.sh.
RECT = {"x": 24, "y": 20, "width": 16, "height": 12}


def build_page() -> tuple[Image.Image, Image.Image]:
    labels = Image.new("L", (WIDTH, HEIGHT), PAPER)
    original = Image.new("RGB", (WIDTH, HEIGHT), (242, 238, 226))

    # Rule band (left) and sky band (right); they touch along column 28/29.
    for y in range(16, 49):
        for x in range(16, 29):
            labels.putpixel((x, y), RULE)
            original.putpixel((x, y), (150, 140, 120))
        for x in range(29, 45):
            labels.putpixel((x, y), SKY)
            original.putpixel((x, y), (47, 93, 138))

    # One-column seal inside the sky band, plus the high-contrast original
    # pixels that make the best unconstrained cuts sit right of the seal.
    for y in range(20, 32):
        labels.putpixel((36, y), SEAL)
        original.putpixel((36, y), (168, 58, 44))
        original.putpixel((35, y), (20, 20, 20))
        original.putpixel((37, y), (10, 10, 10))
        original.putpixel((38, y), (250, 250, 250))
        original.putpixel((39, y), (10, 10, 10))
    return original, labels


def write_pages(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    original, labels = build_page()
    original.save(directory / "page1_rgb.png")
    labels.save(directory / "page1_labels.png")
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "pages": [
                    {
                        "page": 1,
                        "rgb": str(directory / "page1_rgb.png"),
                        "labels": str(directory / "page1_labels.png"),
                        "size": [WIDTH, HEIGHT],
                    }
                ],
                "palette": PALETTE,
                "min_distance": MIN_DISTANCE,
                "rect": RECT,
                "left_label": RULE,
                "right_label": SKY,
                "other_label": SEAL,
                "label_meanings": {
                    str(PAPER): "paper",
                    str(RULE): "rule (left of the boundary)",
                    str(SKY): "sky (right of the boundary)",
                    str(SEAL): "seal (the one other label near the rect)",
                },
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "out", type=Path, nargs="?", default=Path("demo/boundary_pages")
    )
    args = parser.parse_args()
    write_pages(args.out)
    print(f"wrote {args.out}/page1_rgb.png and {args.out}/page1_labels.png")


if __name__ == "__main__":
    main()
