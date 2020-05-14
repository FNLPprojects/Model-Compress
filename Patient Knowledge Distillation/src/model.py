import logging
import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.modeling_bert import BertPreTrainedModel, BertModel, BertEncoder, BertEmbeddings

logger = logging.getLogger(__name__)


class BertForSequenceClassificationEncoder(BertPreTrainedModel):
    def __init__(self, config, num_hidden_layers=None):
        super(BertForSequenceClassificationEncoder, self).__init__(config)
        if num_hidden_layers is not None:
            config.num_hidden_layers = num_hidden_layers

        logger.info("Model config {}".format(config))
        self.bert = BertModel(config)
        self.init_weights()
        self.output_hidden_states = config.output_hidden_states

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        outputs = self.bert(input_ids, token_type_ids, attention_mask)
        if self.output_hidden_states:
            full_output, pooled_output = outputs[2], outputs[1]
            return [full_output[i][:, 0] for i in range(len(full_output)-1)], pooled_output
        else:
            pooled_output = outputs[1]
            return None, pooled_output


class FCClassifierForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, num_labels, hidden_size, n_layers=0):
        super(FCClassifierForSequenceClassification, self).__init__(config)
        self.num_labels = num_labels
        self.n_layers = n_layers
        for i in range(n_layers):
            setattr(self, 'fc%d' % i, nn.Linear(hidden_size, hidden_size))

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.init_weights()

    def forward(self, encoded_feat):
        encoded_feat = self.dropout(encoded_feat)
        for i in range(self.n_layers):
            encoded_feat = getattr(self, 'fc%d' % i)(encoded_feat)
        logits = self.classifier(encoded_feat)
        return logits

