import json
import ray
from ray.tune import register_trainable, run_experiments


def run_distributed_experiment(run_fn, tuna_args, run_args):
    ray.init(num_cpus=tuna_args.num_cpus, num_gpus=tuna_args.num_gpus)

    config = json.load(tuna_args.hyperparameters)
    print(config)

    register_trainable("ray_experiment", run_fn(tuna_args, run_args))

    experiment_spec = {
        tuna_args.experiment_name: {
            "run": "ray_experiment",
            "trial_resources": {
                "cpu": tuna_args.cpus_per_trial,
                "gpu": tuna_args.gpus_per_trial,
            },
            "config": config,
            "local_dir": "./ray_data",
        }
    }

    try:
        print(experiment_spec)
        run_experiments(
            experiments=experiment_spec,
            with_server=tuna_args.with_server,
            server_port=tuna_args.server_port,
        )

    except ray.tune.TuneError as e:
        # TODO: add logging
        print(f"Error during run of experiment '{tuna_args.experiment_name}': {e}")
