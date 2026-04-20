"""
scripts/run_video_inference.py
Batch-process testing videos from S3, write overlaid videos back to S3.

Usage:
  python scripts/run_video_inference.py \
      --model-path /path/to/road_seg_savedmodel \
      --local       # download videos locally first, then process
"""
from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.config_loader import get_config
from src.data.s3_io import download_file, list_keys, upload_file
from src.inference.predictor import LocalPredictor
from src.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)
setup_logging()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True, help="Path to SavedModel dir")
    p.add_argument("--output-prefix", default=None)
    args = p.parse_args()

    cfg     = get_config()
    bucket  = cfg.aws.s3_bucket
    in_pfx  = cfg.aws.s3_prefix.testing
    out_pfx = args.output_prefix or f"{cfg.aws.s3_prefix.artifacts}/video-outputs"

    predictor = LocalPredictor(
        model_path=args.model_path,
        img_size=cfg.data.img_size,
        threshold=cfg.inference.threshold,
    )
    overlay_color = tuple(cfg.inference.overlay_color)  # BGR

    video_keys = [k for k in list_keys(bucket, in_pfx) if k.endswith((".mp4", ".avi", ".mov"))]
    logger.info("Found %d videos in s3://%s/%s", len(video_keys), bucket, in_pfx)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        for key in video_keys:
            name      = Path(key).name
            local_in  = tmpdir / "input"  / name
            local_out = tmpdir / "output" / name

            logger.info("Processing %s ...", name)
            download_file(bucket, key, local_in)
            predictor.process_video(local_in, local_out, overlay_color=overlay_color)

            out_key = f"{out_pfx}/{name}"
            upload_file(local_out, bucket, out_key)
            logger.info("Uploaded → s3://%s/%s", bucket, out_key)

    logger.info("All videos processed.")


if __name__ == "__main__":
    main()
