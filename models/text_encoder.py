# --coding:utf-8--

import torch
from torch import nn
import torch.nn.functional as F
from models.unixcoder import UniXcoder

class TextEncoder(nn.Module):
    def __init__(self, config):
        super(TextEncoder, self).__init__()
        self.config = config
        model_path = config.unixcoder.path
        self.unixcoder = UniXcoder(model_path)
        self.dropout = torch.nn.Dropout(p=config.text_encoder.dropout)

    def forward(self, inputs, seq_lens=None):
        input_ids = self.unixcoder.tokenize(inputs, max_length=1023, padding=True)
        input_ids_tensor = torch.tensor(input_ids).to(next(self.parameters()).device)
        token_embeddings, sentence_embeddings = self.unixcoder(input_ids_tensor)
        sentence_embeddings = self.dropout(sentence_embeddings)
        return sentence_embeddings