import json
import os

from os.path import join, exists
from tuna.runners import AllenNlpRunner
from tuna.executors import RayExecutor


class TestExecutor:
    def test_ray_executor(self, tmpdir, request):
        hyperparam_config_path = join(
            request.fspath.dirname, "fixtures/hyperparam_config.json"
        )
        parameter_config_path = join(
            request.fspath.dirname, "fixtures/parameter_config.jsonnet"
        )
        executor_log_dir = join(tmpdir, "executor_log_dir")
        # serialization_dir = join(tmpdir, "serialization_dir")

        experiment_name = "test_ray_executor"

        arguments = []
        arguments.append("--experiment-name")
        arguments.append(experiment_name)

        arguments.append("--log-dir")
        arguments.append(executor_log_dir)

        arguments.append("--hyperparameters")
        arguments.append(hyperparam_config_path)

        arguments.append("--parameter-file")
        arguments.append(parameter_config_path)

        # arguments.append("--serialization-dir")
        # arguments.append(serialization_dir)

        runner = AllenNlpRunner()
        executor = RayExecutor(runner)

        executor.run(arguments)

        assert exists(executor_log_dir)

        # experiment_dir = join(serialization_dir, experiment_name)
        # assert exists(experiment_dir)

        # allennlp_result_dirs = next(os.walk(experiment_dir))[1]
        # assert len(allennlp_result_dirs) == 4

        # for result_dir in allennlp_result_dirs:
        #     assert exists(join(experiment_dir, result_dir, "model.tar.gz"))

        executor_dir = join(executor_log_dir, experiment_name)
        assert exists(executor_dir)

        executor_result_dirs = next(os.walk(executor_dir))[1]
        for result_dir in executor_result_dirs:
            with open(join(executor_dir, result_dir, "result.json"), "r") as result_f:
                results = json.load(result_f)

            with open(join(executor_dir, result_dir, "params.json"), "r") as params_f:
                params = json.load(params_f)

            assert results["done"]

            trial_dir = join(executor_dir, result_dir, "trial")
            assert exists(trial_dir)

            with open(join(trial_dir, "config.json")) as config_f:
                generated_allennlp_config = json.load(config_f)

            assert (
                params["hidden_size"]
                == generated_allennlp_config["model"]["encoder"]["hidden_size"]
            )
            assert (
                params["num_layers"]
                == generated_allennlp_config["model"]["encoder"]["num_layers"]
            )

            assert exists(join(trial_dir, "model.tar.gz"))
