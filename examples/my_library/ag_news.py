from typing import Dict
import json
import logging

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("ag_news")
class AGNewsDatasetReader(DatasetReader):
    """
    Reads a JSON-lines file containing AG news corpus texts.
    Expected format for each input line: {"text": "text", "label": "label"}
    The output of ``read`` is a list of ``Instance`` s with the fields:
        text: ``TextField``
        label: ``LabelField``
    Parameters
    ----------
    lazy : ``bool`` (optional, default=False)
        Passed to ``DatasetReader``.  If this is ``True``, training will start sooner,
        but will take longer per batch.  This also allows training with datasets that
        are too large to fit in memory.
    tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the title and abstrct into words or other kinds of tokens.
        Defaults to ``WordTokenizer()``.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define input token representations. Defaults to ``{"tokens":
        SingleIdTokenIndexer()}``.
    """

    def __init__(
        self,
        lazy: bool = False,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
    ) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for line in data_file:
                line = line.strip("\n")
                if not line:
                    continue
                news_json = json.loads(line)
                text = news_json["text"]
                label = news_json["label"]
                yield self.text_to_instance(text, label)

    @overrides
    def text_to_instance(
        self, text: str, label: str = None
    ) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        tokenized_text = self._tokenizer.tokenize(text)
        fields = {"text": TextField(tokenized_text, self._token_indexers)}
        if label is not None:
            fields["label"] = LabelField(str(label))
        return Instance(fields)
