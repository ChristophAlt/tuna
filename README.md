# tuna - Hyperparameter search for [AllenNLP](https://github.com/allenai/allennlp), powered by [Ray TUNE](https://ray.readthedocs.io/en/latest/tune.html)

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

## How does it work?
AllenNLP already offers a great way to configure model and training parameter in [Jsonnet](https://jsonnet.org/) (for details, see [Using Config Files](https://github.com/allenai/allennlp/blob/master/tutorials/tagger/README.md#using-config-files)).

Lets assume we configured a simple CNN classifier and want to do hyperparameter search.

```json
[...]
    "model": {
        "type": "cnn-classifier",
        "text_field_embedder": {
            "token_embedders": {
                "tokens": {
                "type": "embedding",
                "embedding_dim": 300,
                },
            }
        },
        "text_encoder": {
            "type": "cnn",
            "embedding_dim": 300,
            "num_filters": 100,
            "ngram_filter_sizes": [2, 3, 4, 5],
        },
        "classifier_feedforward": {
            "input_dim": 400,
            "num_layers": 2,
            "hidden_dims": [200, 2],
            "activations": ["relu", "linear"],
            "dropout": [0.5, 0.0],
        }
    }
[...]
```

It's possible to override the experiment configuration at training time by providing a JSON structure to the train command via the `--overrides` argument. For example, to increase the number of filters of our CNN to 200, we use the following command:

```bash
$ allennlp train <parameter file> \
    --overrides {"model.text_encoder.num_filters": 200}
```

At this point, it looks quite straight forward. The only thing missing is a search strategy (e.g. grid search or random search) to generate the parameter configurations to be evaluated, which than can be used to override the desired parameters at training time. 

Unfortunately, it's not that easy because there are some dependencies between confuration parameters. For example, if we change the number of filters used in our CNN, we must also change the `input_dim` of our `classifier_feedforward` as the output dimension of our CNN is now `num_filters * len(ngram_filter_sizes) = 800`.

Fortunately, Jsonnet provides some nice features around variable substitution. We introduce a local variable `classifier_input_dim`, which we can use to resolve the dependency between `input_dim`, `num_filters`, and `ngram_filter_sizes`.

```json
[...]
local classifier_input_dim = $["model"].text_encoder.num_filters * std.length($["model"].text_encoder.ngram_filter_sizes);

    "model": {
        "type": "cnn-classifier",
        "text_field_embedder": {
            "token_embedders": {
                "tokens": {
                "type": "embedding",
                "embedding_dim": 300,
                },
            }
        },
        "text_encoder": {
            "type": "cnn",
            "embedding_dim": 300,
            "num_filters": 100,
            "ngram_filter_sizes": [2, 3, 4, 5],
        },
        "classifier_feedforward": {
            "input_dim": classifier_input_dim,
            "num_layers": 2,
            "hidden_dims": [200, 2],
            "activations": ["relu", "linear"],
            "dropout": [0.5, 0.0],
        }
    }
[...]
```

Though this seems better, it still doesn't allow us to dynamically configure the parameters, because the local variable `classifier_input_dim` is evaluated when the Jsonnet file is loaded and only then the configuration parameter override happens. Therefore `classifier_input_dim` is evaluated with the default parameters for `num_filters`, and `ngram_filter_sizes`.

Luckily, with Jsonnet we can define the configuration to be a function, taking some arguments with default values and returning a JSON. This has two advantages: the configuration can be used as usual with the `allennlp train` command but most importantly Jsonnet supports the concept of so called [top-level arguments](https://jsonnet.org/learning/tutorial.html) that can be used to parameterize the configuration. This is exactly what tuna does, it combines the functionality of Jsonnet with [Ray TUNE](https://ray.readthedocs.io/en/latest/tune.html) to provide scalable hyperparameter search for AllenNLP.

```json
function (num_filters=100, ngram_filter_sizes=[2, 3, 4, 5]) {

local classifier_input_dim = num_filters * std.length(ngram_filter_sizes);
    [...]
    "model": {
        "type": "cnn-classifier",
        "text_field_embedder": {
            "token_embedders": {
                "tokens": {
                "type": "embedding",
                "embedding_dim": 300,
                },
            }
        },
        "text_encoder": {
            "type": "cnn",
            "embedding_dim": 300,
            "num_filters": num_filters,
            "ngram_filter_sizes": ngram_filter_sizes,
        },
        "classifier_feedforward": {
            "input_dim": classifier_input_dim,
            "num_layers": 2,
            "hidden_dims": [200, 2],
            "activations": ["relu", "linear"],
            "dropout": [0.5, 0.0],
        }
    }
[...]
}
```
