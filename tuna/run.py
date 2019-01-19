#!/usr/bin/env python
import logging
import os

from tuna.runners.allennlp import allennlp_parse_args
from tuna.runners.allennlp import allennlp_train
from tuna.api.run import run

if os.environ.get("TUNA_DEBUG"):
    LEVEL = logging.DEBUG
else:
    LEVEL = logging.INFO

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=LEVEL
)


if __name__ == "__main__":
    run(allennlp_train, allennlp_parse_args)
