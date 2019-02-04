import os
import argparse
import logging
import json
import ray

from ray.tune import register_trainable, run_experiments
from ray.tune.function_runner import StatusReporter
from tuna.executors import Executor
from tuna.runners import Runner

from typing import Dict, Optional, Any, Callable

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class RayExecutor(Executor):
    name = "Ray"

    def __init__(self, runner: Runner) -> None:
        super(RayExecutor, self).__init__(runner)

    def default_argument_parser(self):
        parser = argparse.ArgumentParser(description="Run tuna")

        parser.add_argument(
            "--experiment-name",
            type=str,
            required=True,
            help="a name for the experiment",
        )
        parser.add_argument(
            "--num-cpus",
            type=int,
            default=1,
            help="number of CPUs available to the experiment",
        )
        parser.add_argument(
            "--num-gpus",
            type=int,
            default=1,
            help="number of GPUs available to the experiment",
        )
        parser.add_argument(
            "--cpus-per-trial",
            type=int,
            default=1,
            help="number of CPUs dedicated to a single trial",
        )
        parser.add_argument(
            "--gpus-per-trial",
            type=int,
            default=1,
            help="number of GPUs dedicated to a single trial",
        )
        parser.add_argument(
            "--log-dir",
            type=str,
            default="./logs",
            help="directory in which to store trial logs and results",
        )
        parser.add_argument(
            "--with-server",
            action="store_true",
            default=False,
            help="start the Ray server",
        )
        parser.add_argument(
            "--server-port",
            type=int,
            default=10000,
            help="port for Ray server to listens on",
        )
        parser.add_argument(
            "--search-strategy",
            type=str,
            default="variant-generation",
            help="hyperparameter search strategy used by Ray-Tune",
        )
        parser.add_argument(
            "--hyperparameters",
            type=argparse.FileType("r"),
            required=True,
            help="path to file describing the hyperparameter search space",
        )
        parser.add_argument(
            "--num-samples",
            type=int,
            default=1,
            help="Number of times to sample from the hyperparameter space. "
            + "If grid_search is provided as an argument, the grid will be "
            + "repeated num_samples of times.",
        )

        return parser

    def run_distributed(
        self,
        run_func: Callable[[Dict[str, Any], StatusReporter], None],
        default_args: argparse.Namespace,
        run_args: Optional[argparse.Namespace] = None,
    ) -> None:
        hyperparam_config = json.load(default_args.hyperparameters)
        logger.info(f"Hyperparameter Configuration: {hyperparam_config}")

        logger.info(
            f"Init Ray with {default_args.num_cpus} CPUs "
            + f"and {default_args.num_gpus} GPUs."
        )
        ray.init(num_cpus=default_args.num_cpus, num_gpus=default_args.num_gpus)

        run_func = self._runner.get_run_func(default_args, run_args)
        register_trainable("run", run_func)

        experiments_config = {
            default_args.experiment_name: {
                "run": "run",
                "resources_per_trial": {
                    "cpu": default_args.cpus_per_trial,
                    "gpu": default_args.gpus_per_trial,
                },
                "config": hyperparam_config,
                "local_dir": default_args.log_dir,
                "num_samples": default_args.num_samples,
            }
        }

        logger.info(f"Run Configuration: {experiments_config}")

        try:
            run_experiments(
                experiments=experiments_config,
                with_server=default_args.with_server,
                server_port=default_args.server_port,
            )

        except ray.tune.TuneError as e:
            logger.error(
                f"Error during run of experiment '{default_args.experiment_name}': {e}"
            )
