"""src/utils/logging_utils.py — structured logging setup."""
from __future__ import annotations
import logging
import sys


def setup_logging(level: int = logging.INFO) -> None:
    fmt = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
    logging.basicConfig(
        level=level,
        format=fmt,
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    # Quiet noisy third-party libs
    for lib in ("botocore", "boto3", "urllib3", "s3transfer"):
        logging.getLogger(lib).setLevel(logging.WARNING)
