# tuna - Hyperparameter tuning for [AllenNLP](https://github.com/allenai/allennlp) powered by [Ray](https://github.com/ray-project/ray)

## Usage
Change directory to tuna and use the following command:
```
python -m tuna 
    --experiment-name "<EXPERIMENT NAME>"
    --hyperparameters ./tests/fixtures/hyperparam_config.json 
    --parameter-file ./tests/fixtures/parameter_config.jsonnet 
    --serialization-dir <SERIALIZATION DIR> 
    --num-cpus=4 
    --num-gpus=0 
    --cpus-per-trial=1 
    --gpus-per-trial=0 
    --overrides="{\"train_data_path\":\"<PROJECT DIR>/tests/fixtures/sequence_tagging.tsv\",\"validation_data_path\":\"<PROJECT DIR>/tests/fixtures/sequence_tagging.tsv\"}"
```
