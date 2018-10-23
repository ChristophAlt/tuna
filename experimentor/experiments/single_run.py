from os.path import join

import track


def run_experiment(experiment_fn, default_args, experiment_args):
    local_dir = join(default_args.log_dir, default_args.experiment_name)
    remote_dir = None

    with track.trial(local_dir, remote_dir, param_map=experiment_args):
        track.debug(f"Starting experiment '{default_args.experiment_name}'")
        experiment_fn(experiment_args)
