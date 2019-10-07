from typing import List, Dict, Any, Optional

import os
import logging
import _jsonnet
import json

import torch
from ray import tune
from ray.tune import Experiment, register_trainable
from allennlp.common.params import Params
from allennlp.commands.train import train_model
from allennlp.common.util import import_submodules

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def train_func(
    run_parameters,
    parameter_file_snippet,
    current_working_dir,
    serialization_dir,
    include_packages,
    gpus_available,
):
    def inner(config, reporter):
        logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")

        for package_name in include_packages:
            import_submodules(package_name)

        logger.info(f"Config parameters: {config}")
        logger.info(f"Run parameters: {run_parameters}")

        current_parameters = config or run_parameters
        logger.info(f"Current parameters: {current_parameters}")

        run_params = {k: json.dumps(v) for k, v in current_parameters.items()}
        params_dict = json.loads(
            _jsonnet.evaluate_snippet(
                "config", parameter_file_snippet, tla_codes=run_params
            )
        )

        if not gpus_available:
            logger.warning("No GPU specified, using CPU.")
            params_dict["trainer"]["cuda_device"] = -1

        # Make sure paths are absolute (Ray workers don't use the same working dir)
        for attribute, value in params_dict.items():
            if attribute.endswith("_data_path"):
                path = value
                if not os.path.isabs(path):
                    params_dict[attribute] = os.path.abspath(
                        os.path.join(current_working_dir, path)
                    )

        logger.info(f"AllenNLP Configuration: {params_dict}")
        params = Params(params_dict)

        train_model(params=params, serialization_dir=serialization_dir)

        reporter(done=True)

    return inner


class AllenNlpExperiment:
    def __init__(
        self,
        experiment_name: str,
        parameter_file: str,
        current_working_dir: str,
        log_dir: str,
        resources_per_trial: Dict[str, Any],
        num_samples: Optional[int] = 1,
        run_id: Optional[int] = None,
        serialization_dir: Optional[str] = "./trial",
        run_parameters: Optional[Dict[str, Any]] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        include_packages: Optional[List[str]] = None,
    ) -> None:
        if run_parameters is not None and hyperparameters is not None:
            raise ValueError(
                f"'run_parameters' and 'hyperparameters' can't be both specified."
            )

        self._experiment_name = experiment_name
        self._parameter_file = parameter_file
        self._current_working_dir = current_working_dir
        self._log_dir = log_dir
        self._resources_per_trial = resources_per_trial
        self._gpus_available = (
            "gpu" in resources_per_trial and resources_per_trial["gpu"] > 0
        )
        self._run_id = run_id
        self._serialization_dir = serialization_dir
        self._run_parameters = run_parameters or {}
        self._hyperparameters = hyperparameters or {}
        self._include_packages = include_packages or []
        if "cpu" in resources_per_trial and resources_per_trial["cpu"] > 0:
            num_pytorch_threads = resources_per_trial["cpu"]
            logger.debug(f"Limiting pytorch to {num_pytorch_threads} threads")
            torch.set_num_threads(num_pytorch_threads)

    def to_ray_experiment(self) -> Experiment:
        with open(self._parameter_file, "r") as parameter_f:
            parameter_file_snippet = parameter_f.read()

        trainable_name = f"{self._experiment_name}_train_func_{self._run_id or '0'}"

        logger.info(f"Register run parameters: {self._run_parameters}")

        register_trainable(
            trainable_name,
            train_func(
                run_parameters=self._run_parameters,
                parameter_file_snippet=parameter_file_snippet,
                current_working_dir=self._current_working_dir,
                serialization_dir=self._serialization_dir,
                include_packages=self._include_packages,
                gpus_available=self._gpus_available,
            ),
        )

        def trial_name_creator(trial):
            params = ",".join(
                [
                    f"{k}={v}"
                    for k, v in sorted(
                        self._run_parameters.items(), key=lambda kv: kv[0]
                    )
                ]
            )
            return f"{trial}_{params}"

        config = self._hyperparameters or {}
        return Experiment(
            name=self._experiment_name,
            run=trainable_name,
            config=config,
            resources_per_trial=self._resources_per_trial,
            local_dir=self._log_dir,
            trial_name_creator=tune.function(trial_name_creator)
            if self._run_parameters
            else None,
        )
