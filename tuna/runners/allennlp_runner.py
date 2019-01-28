import os
import argparse
import logging
import _jsonnet
import json

from datetime import datetime
from allennlp.common.params import Params, parse_overrides, with_fallback
from allennlp.commands.train import train_model
from allennlp.common.util import import_submodules
from tuna.runners import Runner

from typing import Optional

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class AllenNlpRunner(Runner):
    name = "AllenNLP"

    def get_argument_parser(self) -> Optional[argparse.ArgumentParser]:
        parser = argparse.ArgumentParser()

        parser.add_argument(
            "--parameter-file",
            required=True,
            type=os.path.abspath,
            help="path to parameter file describing the model to be trained",
        )
        # parser.add_argument(
        #     "-s",
        #     "--serialization-dir",
        #     required=True,
        #     type=str,
        #     help="directory in which to save the model and its logs",
        # )
        parser.add_argument(
            "--include-package",
            type=str,
            action="append",
            default=[],
            help="additional packages to include",
        )
        parser.add_argument(
            "-o",
            "--overrides",
            type=str,
            default="",
            help="a JSON structure used to override the experiment configuration",
        )
        return parser

    def get_run_func(
        self,
        default_args: argparse.Namespace,
        run_args: Optional[argparse.Namespace] = None,
    ):
        if run_args is None:
            raise ValueError("No run arguments found for AllenNLP runner.")

        with open(run_args.parameter_file, "r") as parameter_f:
            parameter_file_snippet = parameter_f.read()

        def train_func(config, reporter):
            logger.debug(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")

            for package_name in getattr(run_args, "include_package", ()):
                import_submodules(package_name)

            run_parameters = {k: json.dumps(v) for k, v in config.items()}

            file_dict = json.loads(
                _jsonnet.evaluate_snippet(
                    "config", parameter_file_snippet, tla_codes=run_parameters
                )
            )
            if default_args.num_gpus == 0:
                logger.warning(f"No GPU specified, using CPU.")
                file_dict["trainer"]["cuda_device"] = -1

            overrides_dict = parse_overrides(run_args.overrides)

            params_dict = with_fallback(preferred=overrides_dict, fallback=file_dict)

            # Make sure path is absolute (as Ray workers do not use the same working dir)
            params_dict["train_data_path"] = os.path.abspath(
                os.path.join(
                    os.path.dirname(run_args.parameter_file),
                    params_dict["train_data_path"],
                )
            )
            if "validation_data_path" in params_dict:
                params_dict["validation_data_path"] = os.path.abspath(
                    os.path.join(
                        os.path.dirname(run_args.parameter_file),
                        params_dict["validation_data_path"],
                    )
                )

            params = Params(params_dict)

            logger.debug(f"AllenNLP Configuration: {params.as_dict()}")

            # serialization_dir = os.path.join(
            #     run_args.serialization_dir,
            #     default_args.experiment_name,
            #     datetime.now().strftime("%Y-%m-%d__%H-%M__%f"),
            # )

            train_model(params=params, serialization_dir="./trial/")

            reporter(done=True)

        return train_func
