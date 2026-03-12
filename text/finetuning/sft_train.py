#!/usr/bin/env python3
"""
Apertus-style entrypoint:

accelerate launch --config_file configs/zero3.yaml \
  sft_train.py --config configs/sft_full.yaml
"""

import os
import sys

from train import get_args, main, set_seed
from transformers import logging


def _translate_argv(argv):
    translated = []
    skip_next = False
    for i, token in enumerate(argv):
        if skip_next:
            skip_next = False
            continue
        if token == "--config":
            translated.append("--config_file")
            continue
        if token.startswith("--config="):
            translated.append(token.replace("--config=", "--config_file=", 1))
            continue
        translated.append(token)
    return translated


if __name__ == "__main__":
    sys.argv = _translate_argv(sys.argv)
    args = get_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    logging.set_verbosity_error()
    main(args)
