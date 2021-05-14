import torch
from torch import nn
from transformers import AutoModel, AutoConfig


class RE(nn.Module):

    def __init__(self, pretrain_path="scibert-scivocab-cased", num_labels=8, num_hidden_layers=12):
        super(RE, self).__init__()
        self.pretrain_path = pretrain_path
        config = AutoConfig.from_pretrained(pretrain_path, mirror='tuna')
        config.num_hidden_layers = num_hidden_layers
        self.bert = AutoModel.from_pretrained(pretrain_path, config=config, mirror='tuna')
        self.classifier = nn.Linear(2*config.hidden_size, num_labels)
        self.cost = nn.CrossEntropyLoss()

    def forward(self, inputs):
        # (B, L, H)
        output = self.bert(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])[0]
        offset = 1  # [CLS] is prepended to the sequence
        # head (B, H); indices (B,)
        # head = batched_index_select(output, indices=inputs['h_pos'] + offset)
        head = torch.stack([output[i][index+offset] for i, index in enumerate(inputs['h_pos'])], dim=0)
        # tail (B, H)
        # tail = batched_index_select(output, indices=inputs['t_pos'] + offset)
        tail = torch.stack([output[i][index+offset] for i, index in enumerate(inputs['t_pos'])], dim=0)
        # (B, 2H)
        entity_pair = torch.cat([head, tail], dim=-1)
        # (B, num_labels)
        logits = self.classifier(entity_pair)
        return logits

    def loss(self, logits, label):
        N = logits.size(-1)
        return self.cost(logits.view(-1, N), label.view(-1))




