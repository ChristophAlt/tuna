import _jsonnet
import json
import os
from datetime import datetime
from allennlp.common import Params
from allennlp.commands.train import train_model
from allennlp.common.util import import_submodules


def allennlp_parse_args(parser):

    parser.add_argument(
        "--parameter-file",
        required=True,
        type=str,
        help="path to parameter file describing the model to be trained",
    )
    parser.add_argument(
        "-s",
        "--serialization-dir",
        required=True,
        type=str,
        help="directory in which to save the model and its logs",
    )
    parser.add_argument(
        "--include-package",
        type=str,
        action="append",
        default=[],
        help="additional packages to include",
    )


def allennlp_train(
    tuna_args,
    run_args,
    # parameter_file: str,
    # serialization_dir: str,
    # file_friendly_logging: bool = False,
    # recover: bool = False,
    # force: bool = False,
) -> None:
    with open(run_args.parameter_file, "r") as parameter_f:
        parameter_file_snippet = parameter_f.read()

    def inner_fn(config, reporter):
        for package_name in getattr(run_args, "include_package", ()):
            import_submodules(package_name)

        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
        run_parameters = {k: json.dumps(v) for k, v in config.items()}
        params_dict = json.loads(
            _jsonnet.evaluate_snippet(
                "snippet", parameter_file_snippet, tla_codes=run_parameters
            )
        )
        params = Params(params_dict)
        print(params.as_dict())

        serialization_dir = os.path.join(
            run_args.serialization_dir,
            tuna_args.experiment_name,
            datetime.now().strftime("%Y-%m-%d__%H-%M__%f"),
        )

        train_model(params=params, serialization_dir=serialization_dir)

        reporter(done=True, serialization_dir=serialization_dir)

    return inner_fn

    # for package_name in getattr(runner_args, "include_package", ()):
    #     import_submodules(package_name)

    # del runner_args.include_package


# def train_model(params: Params,
#                 serialization_dir: str,
#                 file_friendly_logging: bool = False,
#                 recover: bool = False,
#                 force: bool = False) -> Model:
