from typing import List, Union, Optional

import logging
import ray
from ray.tune import run_experiments
from tuna.allennlp_experiment import AllenNlpExperiment


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class RayExecutor:
    def __init__(
        self, num_cpus: int, num_gpus: int, server_port: Optional[int] = None
    ) -> None:
        self._ray = ray.init(num_cpus=num_cpus, num_gpus=num_gpus)

        logger.info(f"Init Ray with {num_cpus} CPUs and {num_gpus} GPUs.")

    def run(self, experiments: Union[AllenNlpExperiment, List[AllenNlpExperiment]]):
        if type(experiments) is not list:
            experiments = [experiments]

        run_experiments([experiment.to_ray_experiment() for experiment in experiments])
