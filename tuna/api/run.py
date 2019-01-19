import argparse
from tuna.experiments import run_distributed_experiment


def _default_argument_parser():
    parser = argparse.ArgumentParser(description="tuna")

    parser.add_argument("--experiment-name", type=str, required=True)
    parser.add_argument("--num-cpus", type=int, default=1)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--cpus-per-trial", type=int, default=1)
    parser.add_argument("--gpus-per-trial", type=int, default=1)
    parser.add_argument("--log-dir", type=str, default="./logs")

    parser.add_argument("--with-server", action="store_true", default=False)
    parser.add_argument("--server-port", type=int, default=10000)

    parser.add_argument("--search-strategy", type=str, default="variant-generation")
    parser.add_argument("--hyperparameters", type=argparse.FileType("r"), required=True)

    return parser


def run(run_fn, run_args_fn=None):
    parser = _default_argument_parser()
    tuna_args, remaining_args = parser.parse_known_args()

    if run_args_fn:
        parser = argparse.ArgumentParser()
        run_args_fn(parser)

    run_args = parser.parse_args(remaining_args)

    print(tuna_args)
    print(run_args)

    run_distributed_experiment(run_fn, tuna_args, run_args)
