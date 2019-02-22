from typing import List, Dict, Any, Optional

import os
import argparse
import logging
import _jsonnet
import json
from functools import partial
from ray.tune import Experiment, register_trainable
from allennlp.common.params import Params
from allennlp.commands.train import train_model
from allennlp.common.util import import_submodules


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class AllenNlpExperiment:
    def __init__(
        self,
        experiment_name: str,
        parameter_file: str,
        current_working_dir: str,
        log_dir: str,
        resources_per_trial: Dict[str, Any],
        num_samples: Optional[int] = 1,
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
        self._serialization_dir = serialization_dir
        self._run_parameters = run_parameters or {}
        self._hyperparameters = hyperparameters or {}
        self._include_packages = include_packages or []

    def to_ray_experiment(self,) -> Experiment:
        with open(self._parameter_file, "r") as parameter_f:
            parameter_file_snippet = parameter_f.read()

        def train_func(
            config,
            reporter,
            current_working_dir,
            serialization_dir,
            include_packages,
            gpus_available,
            run_parameters=None,
        ):
            logger.debug(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")

            for package_name in include_packages:
                import_submodules(package_name)

            run_parameters = config or run_parameters

            run_params = {k: json.dumps(v) for k, v in run_parameters.items()}
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

            logger.debug(f"AllenNLP Configuration: {params_dict}")
            params = Params(params_dict)

            train_model(params=params, serialization_dir=serialization_dir)

            reporter(done=True)

        trainable_name = f"{self._experiment_name}_train_func"
        register_trainable(
            trainable_name,
            partial(
                train_func,
                current_working_dir=self._current_working_dir,
                serialization_dir=self._serialization_dir,
                include_packages=self._include_packages,
                gpus_available=self._gpus_available,
                run_parameters=self._run_parameters,
            ),
        )

        config = self._hyperparameters or {}
        return Experiment(
            name=self._experiment_name,
            run=trainable_name,
            config=config,
            resources_per_trial=self._resources_per_trial,
            local_dir=self._log_dir,
        )
