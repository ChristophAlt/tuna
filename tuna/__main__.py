#!/usr/bin/env python
import sys
import os
import json
import argparse
import logging

from tuna.ray_executor import RayExecutor
from tuna.allennlp_experiment import AllenNlpExperiment


if os.environ.get("TUNA_DEBUG"):
    LEVEL = logging.DEBUG
else:
    LEVEL = logging.INFO

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=LEVEL
)


parser = argparse.ArgumentParser(description="Run distributed experiments with tuna")
parser.add_argument(
    "--num-cpus",
    type=int,
    default=1,
    help="number of CPUs available to tuna experiments",
)
parser.add_argument(
    "--num-gpus",
    type=int,
    default=1,
    help="number of GPUs available to tuna experiments",
)

subparsers = parser.add_subparsers()

parser_single = subparsers.add_parser("single")
parser_single.add_argument(
    "--experiment-name", type=str, required=True, help="a name for the experiment"
)
parser_single.add_argument(
    "--parameter-file",
    required=True,
    type=os.path.abspath,
    help="path to parameter file describing the model to be trained",
)
parser_single.add_argument(
    "--hyperparameters",
    type=os.path.abspath,
    required=True,
    help="path to file describing the hyperparameter search space",
)
parser_single.add_argument(
    "--include-package",
    type=str,
    action="append",
    default=[],
    help="additional packages to include",
)
parser_single.add_argument(
    "--cpus-per-trial",
    type=int,
    default=1,
    help="number of CPUs dedicated to a single trial",
)
parser_single.add_argument(
    "--gpus-per-trial",
    type=int,
    default=1,
    help="number of GPUs dedicated to a single trial",
)
parser_single.add_argument(
    "--log-dir",
    type=str,
    default="./logs",
    help="directory in which to store trial logs and results",
)
parser_single.add_argument(
    "--with-server", action="store_true", default=False, help="start the Ray server"
)
parser_single.add_argument(
    "--server-port", type=int, default=4321, help="port for Ray server to listens on"
)
parser_single.add_argument(
    "--search-strategy",
    type=str,
    default="variant-generation",
    help="hyperparameter search strategy used by Ray-Tune",
)
parser_single.add_argument(
    "--num-samples",
    type=int,
    default=1,
    help="Number of times to sample from the hyperparameter space. "
    + "If grid_search is provided as an argument, the grid will be "
    + "repeated num_samples of times.",
)


def single(args):
    with open(args.hyperparameters, "r") as hyperparameter_f:
        hyperparameters = json.load(hyperparameter_f)

    executor = RayExecutor(num_cpus=args.num_cpus, num_gpus=args.num_gpus)

    resources_per_trial = {"cpu": args.cpus_per_trial, "gpu": args.gpus_per_trial}
    experiment = AllenNlpExperiment(
        experiment_name=args.experiment_name,
        parameter_file=args.parameter_file,
        hyperparameters=hyperparameters,
        current_working_dir=os.getcwd(),
        log_dir=args.log_dir,
        resources_per_trial=resources_per_trial,
        num_samples=args.num_samples,
        include_packages=args.include_package,
    )

    executor.run(experiment)


parser_multi = subparsers.add_parser("multi")
parser_multi.add_argument(
    "--run-file",
    required=True,
    type=os.path.abspath,
    help=(
        "path to run file describing the experiments as a combination of "
        + "parameter file and hyperparameters or run parameters"
    ),
)
parser_multi.add_argument(
    "--include-package",
    type=str,
    action="append",
    default=[],
    help="additional packages to include",
)
parser_multi.add_argument(
    "--cpus-per-trial",
    type=int,
    default=1,
    help="number of CPUs dedicated to a single trial",
)
parser_multi.add_argument(
    "--gpus-per-trial",
    type=int,
    default=1,
    help="number of GPUs dedicated to a single trial",
)
parser_multi.add_argument(
    "--log-dir",
    type=str,
    default="./logs",
    help="directory in which to store trial logs and results",
)
parser_multi.add_argument(
    "--with-server", action="store_true", default=False, help="start the Ray server"
)
parser_multi.add_argument(
    "--server-port", type=int, default=4321, help="port for Ray server to listens on"
)


def multi(args):
    with open(args.run_file, "r") as run_f:
        run_config = json.load(run_f)

    resources_per_trial = {"cpu": args.cpus_per_trial, "gpu": args.gpus_per_trial}

    experiments = []
    for run in run_config["runs"]:
        experiment_name = run["experiment_name"]
        parameter_file = run["parameter_file"]

        hyperparameters = run.get("hyperparameters")
        num_samples = run.get("num_samples", 1)
        # if hyperparameters:
        #     with open(hyperparameters, "r") as hyperparameter_f:
        #         hyperparameters = json.load(hyperparameter_f)

        run_parameters = run.get("run_parameters")

        if run_parameters:
            for run_id, run_p in enumerate(run_parameters):
                experiment = AllenNlpExperiment(
                    experiment_name=experiment_name,
                    parameter_file=parameter_file,
                    run_parameters=run_p,
                    current_working_dir=os.getcwd(),
                    log_dir=args.log_dir,
                    resources_per_trial=resources_per_trial,
                    num_samples=num_samples,
                    include_packages=args.include_package,
                    run_id=run_id,
                )
                experiments.append(experiment)
        else:
            experiment = AllenNlpExperiment(
                experiment_name=experiment_name,
                parameter_file=parameter_file,
                hyperparameters=hyperparameters,
                current_working_dir=os.getcwd(),
                log_dir=args.log_dir,
                resources_per_trial=resources_per_trial,
                num_samples=num_samples,
                include_packages=args.include_package,
            )
            experiments.append(experiment)

    executor = RayExecutor(num_cpus=args.num_cpus, num_gpus=args.num_gpus)
    executor.run(experiments)


def main():
    args = parser.parse_args()
    args.func(args)


parser_single.set_defaults(func=single)
parser_multi.set_defaults(func=multi)


if __name__ == "__main__":
    main()
