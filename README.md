# tuna - Hyperparameter search for [AllenNLP](https://github.com/allenai/allennlp), powered by [Ray](https://github.com/ray-project/ray)

## Installation

### With pip
tuna can be installed by pip as follows:
```bash
TBD
```

### From source

Clone the repository and run:
```bash
pip install [--editable] .
```

A series of tests is included in the [tests folder](https://github.com/ChristophAlt/tuna/tree/master/tests).

You can run the tests with the command (install pytest if needed: `pip install pytest`):
```bash
pytest -vv
```

## Running tuna
```bash
$ tuna --help
Run tuna

optional arguments:
  -h, --help            show this help message and exit
  --experiment-name EXPERIMENT_NAME
                        a name for the experiment
  --num-cpus NUM_CPUS   number of CPUs available to the experiment
  --num-gpus NUM_GPUS   number of GPUs available to the experiment
  --cpus-per-trial CPUS_PER_TRIAL
                        number of CPUs dedicated to a single trial
  --gpus-per-trial GPUS_PER_TRIAL
                        number of GPUs dedicated to a single trial
  --log-dir LOG_DIR     directory in which to store trial logs and results
  --with-server         start the Ray server
  --server-port SERVER_PORT
                        port for Ray server to listens on
  --search-strategy SEARCH_STRATEGY
                        hyperparameter search strategy used by Ray-Tune
  --hyperparameters HYPERPARAMETERS
                        path to file describing the hyperparameter search
                        space
```

### Example

Switch to tuna directory and run the following command:
```bash
$ tuna \
    --experiment-name test_experiment \
    --hyperparameters ./tests/fixtures/hyperparam_config.json \
    --parameter-file ./tests/fixtures/parameter_config.jsonnet \
    --log-dir ./experiments/ \
    --num-cpus=2 \
    --num-gpus=0 \
    --cpus-per-trial=1 \
    --gpus-per-trial=0
```
