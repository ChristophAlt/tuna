import ray
import yaml
from ray.tune import register_trainable, run_experiments
from experimentor.experiments.single_run import run_experiment


def _ray_experiment_fn(experiment_fn, default_args, experiment_args):

    #for k, v in config.items():
    #    setattr(experiment_args, k, v)

    def inner_func(config, reporter):
        updated_experiment_args = vars(experiment_args)
        updated_experiment_args.update(dict(config.items()))

        run_experiment(experiment_fn, default_args, updated_experiment_args)
        reporter(done=True)

    return inner_func


def run_distributed_experiment(experiment_fn, default_args, experiment_args):
    ray.init(num_cpus=default_args.num_cpus, num_gpus=default_args.num_gpus)

    config = yaml.load(default_args.with_config)
    print(config)

    register_trainable('ray_experiment', _ray_experiment_fn(
        experiment_fn, default_args, experiment_args))

    experiment_spec = {
        default_args.experiment_name: {
            'run': 'ray_experiment',
            'trial_resources': {
                'cpu': default_args.num_cpus_per_trial,
                'gpu': default_args.num_gpus_per_trial
            },
            'config': config,
            'local_dir': './ray_data'
        }
    }

    try:
        run_experiments(
            experiments=experiment_spec,
            with_server=default_args.with_server,
            server_port=default_args.server_port)

    except ray.tune.TuneError as e:
        # TODO: add logging
        print(f"Error during run of experiment '{default_args.experiment_name}': {e}")
    