"""
Small CLI that mirrors test_deeptag.py behavior.
"""

import argparse
import cv2
import sys
from deeptag_pytorch.core import DeepTag

def main(argv=None):
    parser = argparse.ArgumentParser(prog="deeptag-test")
    parser.add_argument("image", help="path to image to run detection on")
    parser.add_argument("--model-dir", default=None, help="path to folder containing model .pth files (overrides bundled models/)")
    parser.add_argument("--family", default="aruco")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args(argv)

    img = cv2.imread(args.image)
    if img is None:
        print("Failed to read image:", args.image)
        return 2

    device = 'cpu' if args.cpu else None
    dt = DeepTag(tag_family=args.family, device=device, model_dir=args.model_dir)
    results = dt.predict(img)
    print("Results:", results)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
