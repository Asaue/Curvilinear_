#!/usr/bin/env python3
"""Convert image2label JSON into the label2image_test.json format used by test.py."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image2label', required=True, help='Path to image2label_train.json')
    parser.add_argument('--out', required=True, help='Output label2image_test.json path')
    args = parser.parse_args()

    src = Path(args.image2label)
    dst = Path(args.out)
    with src.open('r', encoding='utf-8') as f:
        image2label = json.load(f)

    label2image = {label: image for image, label in image2label.items()}
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open('w', encoding='utf-8') as f:
        json.dump(label2image, f, indent=2, ensure_ascii=False)
    print(f'Wrote {len(label2image)} pairs to {dst}')


if __name__ == '__main__':
    main()
