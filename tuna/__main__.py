#!/usr/bin/env python
import sys
import logging
import os

from tuna.runners import AllenNlpRunner
from tuna.executors import RayExecutor

if os.environ.get("TUNA_DEBUG"):
    LEVEL = logging.DEBUG
else:
    LEVEL = logging.INFO

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=LEVEL
)


def main():
    runner = AllenNlpRunner()
    executor = RayExecutor(runner)
    executor.run(sys.argv[1:])


if __name__ == "__main__":
    main()
