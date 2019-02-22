function (hidden_size=7, num_layers=3) {
    "model": {
        "type": "simple_tagger",
        "text_field_embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "embedding_dim": 5
                }
            }
        },
        "encoder": {
            "type": "lstm",
            "input_size": 5,
            "hidden_size": hidden_size,
            "num_layers": num_layers
        }
    },
    "dataset_reader": {"type": "sequence_tagging"},
    "train_data_path": "./tests/fixtures/sequence_tagging.tsv",
    "validation_data_path": "./tests/fixtures/sequence_tagging.tsv",
    "iterator": {"type": "basic", "batch_size": 2},
    "trainer": {
        "num_epochs": 2,
        "optimizer": "adam"
    }
}