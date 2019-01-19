#!/usr/bin/env python
import sys
import logging
import os

from tuna.runners import AllenNlpRunner
from tuna.executors import RayExecutor

# from tuna.runners.allennlp import allennlp_parse_args
# from tuna.runners.allennlp import allennlp_train
# from tuna.api.run import run

if os.environ.get("TUNA_DEBUG"):
    LEVEL = logging.DEBUG
else:
    LEVEL = logging.INFO

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=LEVEL
)


if __name__ == "__main__":
    runner = AllenNlpRunner()
    executor = RayExecutor(runner)
    executor.run(sys.argv[1:])
