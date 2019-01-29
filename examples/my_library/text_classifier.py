from typing import Dict, Optional

import numpy
from overrides import overrides
import torch
import torch.nn.functional as F

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import FeedForward, Seq2VecEncoder, TextFieldEmbedder
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy


@Model.register("text_classifier")
class TextClassifier(Model):
    """
    This ``Model`` assumes a given text and performs text classification on it,
    predicting some output label. The basic model structure: we'll embed the text
    and encode it with a Seq2VecEncoder, getting a single vector representing the
    content.  We'll then and pass the vector through a feedforward network,
    the output of which we'll use as our scores for each label.
    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : ``TextFieldEmbedder``, required
        Used to embed the ``tokens`` ``TextField`` we get as input to the model.
    text_encoder : ``Seq2VecEncoder``
        The encoder that we will use to convert the text to a vector.
    classifier_feedforward : ``FeedForward``
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        text_encoder: Seq2VecEncoder,
        classifier_feedforward: FeedForward,
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
    ) -> None:
        super(TextClassifier, self).__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.num_classes = self.vocab.get_vocab_size("labels")
        self.text_encoder = text_encoder
        self.classifier_feedforward = classifier_feedforward

        if text_field_embedder.get_output_dim() != text_encoder.get_input_dim():
            raise ConfigurationError(
                "The output dimension of the text_field_embedder must match the "
                "input dimension of the text_encoder. Found {} and {}, "
                "respectively.".format(
                    text_field_embedder.get_output_dim(), text_encoder.get_input_dim()
                )
            )

        self.metrics = {"accuracy": CategoricalAccuracy()}

        self.loss = torch.nn.CrossEntropyLoss()

        initializer(self)

    @overrides
    def forward(
        self,  # type: ignore
        text: Dict[str, torch.LongTensor],
        label: torch.LongTensor = None,
    ) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        text : Dict[str, Variable], required
            The output of ``TextField.as_array()``.
        label : Variable, optional (default = None)
            A variable representing the label for each instance in the batch.
        Returns
        -------
        An output dictionary consisting of:
        class_probabilities : torch.FloatTensor
            A tensor of shape ``(batch_size, num_classes)`` representing a
            distribution over the label classes for each instance.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """
        embedded_text = self.text_field_embedder(text)
        text_mask = util.get_text_field_mask(text)
        encoded_text = self.text_encoder(embedded_text, text_mask)

        logits = self.classifier_feedforward(encoded_text)
        output_dict = {"logits": logits}
        if label is not None:
            loss = self.loss(logits, label)
            for metric in self.metrics.values():
                metric(logits, label)
            output_dict["loss"] = loss

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does a simple argmax over the class probabilities, converts indices to string labels, and
        adds a ``"label"`` key to the dictionary with the result.
        """
        class_probabilities = F.softmax(output_dict["logits"], dim=-1)
        output_dict["class_probabilities"] = class_probabilities

        predictions = class_probabilities.cpu().data.numpy()
        argmax_indices = numpy.argmax(predictions, axis=-1)
        labels = [
            self.vocab.get_token_from_index(x, namespace="labels")
            for x in argmax_indices
        ]
        output_dict["label"] = labels
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            metric_name: metric.get_metric(reset)
            for metric_name, metric in self.metrics.items()
        }
