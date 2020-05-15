# coding=utf-8
import os
import torch
from torch import nn
from transformers import BertPreTrainedModel, BertModel, BertConfig
from transformers.modeling_bert import BertPreTrainingHeads
CONFIG_NAME = 'config.json'

class TinyBertForPreTraining(BertPreTrainedModel):
    def __init__(self, config, fit_size=768):
        super(TinyBertForPreTraining, self).__init__(config)
        config.output_attentions = True
        config.output_hidden_states = True
        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(config)
        self.fit_dense = nn.Linear(config.hidden_size, fit_size)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None,
                attention_mask=None, masked_lm_labels=None,
                next_sentence_label=None, labels=None):
        outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        sequence_output = outputs[2]
        att_output = outputs[3]
        pooled_output = outputs[1]
        tmp = []
        for s_id, sequence_layer in enumerate(sequence_output):
            tmp.append(self.fit_dense(sequence_layer))
        sequence_output = tmp

        return att_output, sequence_output

    @classmethod
    def from_scratch(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        resolved_config_file = os.path.join(
            pretrained_model_name_or_path)
        config = BertConfig.from_json_file(resolved_config_file)

        model = cls(config, *inputs, **kwargs)
        return model

class TinyBertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, fit_size=768):
        super(TinyBertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        config.output_attentions = True
        config.output_hidden_states = True
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.fit_dense = nn.Linear(config.hidden_size, fit_size)

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                labels=None, is_student=False):

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[2]
        att_output = outputs[3]
        pooled_output = outputs[1]

        # print('sequence_output:',len(sequence_output))
        # print('att_output:',len(att_output))
        # print('pooled_output:',pooled_output.size())
        # sequence_output: 5
        # att_output: 4
        # pooled_output: torch.Size([32, 312])
        # sequence_output: 13
        # att_output: 12
        # pooled_output: torch.Size([32, 768])

        logits = self.classifier(torch.relu(pooled_output))

        tmp = []
        if is_student:
            for s_id, sequence_layer in enumerate(sequence_output):
                tmp.append(self.fit_dense(sequence_layer))
            sequence_output = tmp
        return logits, att_output, sequence_output
