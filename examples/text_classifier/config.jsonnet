function(num_filters=5, ngram_filter_sizes=[2], batch_size=1, num_epochs=5) {

    local classifier_input_dim = num_filters * std.length(ngram_filter_sizes),

    "train_data_path": "./ag_news_corpus.jsonl",

    "dataset_reader": {
        "type": "ag_news",
        "token_indexers": {
            "tokens": {
                "type": "single_id",
                "lowercase_tokens": true,
            },
        }
    },

    "model": {
        "type": "text_classifier",
        "text_field_embedder": {
            "token_embedders": {
                "tokens": {
                "type": "embedding",
                "embedding_dim": 10,
                },
            }
        },
        "text_encoder": {
            "type": "cnn",
            "embedding_dim": 10,
            "num_filters": num_filters,
            "ngram_filter_sizes": ngram_filter_sizes,
        },
        "classifier_feedforward": {
            "input_dim": classifier_input_dim,
            "num_layers": 2,
            "hidden_dims": [5, 2],
            "activations": ["relu", "linear"],
            "dropout": [0.5, 0.0],
        }
    },

    "iterator": {
        "type": "bucket",
        "sorting_keys": [["text", "num_tokens"]],
        "batch_size": batch_size,
    },

    "trainer": {
        "num_epochs": num_epochs,
        "cuda_device": 0,
        "validation_metric": "+accuracy",
        "optimizer": {
            "type": "sgd",
            "lr": 1e-3
        }
    }
}
