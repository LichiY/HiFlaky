# --coding:utf-8--
#!/usr/bin/env python
# coding:utf-8


import torch
from models.unixcoder import UniXcoder


class EmbeddingLayer(torch.nn.Module):
    def __init__(self,
                 embedding_dim,
                 vocab_name,
                 config,
                 ):
        super(EmbeddingLayer, self).__init__()

        model_path = config.unixcoder.path
        self.embedding = UniXcoder(model_path)
        for param in self.embedding.parameters():
            param.requires_grad = False
        self.embedding_dim = embedding_dim
        self.dropout = torch.nn.Dropout(p=config['embedding'][vocab_name]['dropout'])

    def forward(self, vocab_id_list):
        token_embeddings, sentence_embeddings = self.embedding(vocab_id_list)
        return token_embeddings, sentence_embeddings
