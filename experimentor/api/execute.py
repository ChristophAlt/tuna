import argparse

from experimentor.experiments import run_experiment, run_distributed_experiment


def _default_argument_parser():
    parser = argparse.ArgumentParser(description='Experimentor')

    parser.add_argument('experiment_name', type=str)
    parser.add_argument('--num_cpus', type=int, default=1)
    parser.add_argument('--num_gpus', type=int, default=0)
    parser.add_argument('--num_cpus_per_trial', type=int, default=1)
    parser.add_argument('--num_gpus_per_trial', type=int, default=0)
    parser.add_argument('--with_server', action='store_true', default=False)
    parser.add_argument('--server_port', type=int, default=10000)

    parser.add_argument('--with_config', type=argparse.FileType('r'), default=None)

    parser.add_argument('--log_dir', type=str, default='./logs')

    return parser


def execute(experiment_fn, experiment_args_fn=None):
    # TODO: check if we can make everything configurable by yaml file

    parser = _default_argument_parser()
    default_args, remaining_args = parser.parse_known_args()

    if experiment_args_fn:
        parser = argparse.ArgumentParser()
        experiment_args_fn(parser)

    experiment_args = parser.parse_args(remaining_args)

    print(default_args)
    print(experiment_args)

    if default_args.with_config is None:
        run_experiment(experiment_fn, default_args, vars(experiment_args))
    else:
        run_distributed_experiment(experiment_fn, default_args, experiment_args)
