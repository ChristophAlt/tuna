from tuna.runners import AllenNlpRunner


class TestRunner:
    def test_allennlp_runner_arg_parse(self):
        runner = AllenNlpRunner()

        assert runner.name == "AllenNLP"

        parser = runner.get_argument_parser()
        run_args = parser.parse_args(
            [
                "--parameter-file",
                "/parameter/file.jsonnet",
                "--include-package",
                "package1",
            ]
        )

        run_args_dict = vars(run_args)
        assert run_args_dict["parameter_file"] == "/parameter/file.jsonnet"
        assert run_args_dict["include_package"] == ["package1"]
