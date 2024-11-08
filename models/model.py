# --coding:utf-8--


import torch.nn as nn
from models.structure_model.structure_encoder import StructureEncoder
from models.text_encoder import TextEncoder
from models.embedding_layer import EmbeddingLayer
from models.text_feature_propagation import Propagation



class HiFlaky(nn.Module):
    def __init__(self, config, vocab, model_type, model_mode='TRAIN'):
        super(HiFlaky, self).__init__()
        self.config = config
        self.vocab = vocab
        self.device = config.train.device_setting.device
        self.token_map, self.label_map = vocab.v2i['token'], vocab.v2i['label']
        self.token_embedding = EmbeddingLayer(
            embedding_dim=config.embedding.token.dimension,
            vocab_name='token',
            config=config,
        )

        self.text_encoder = TextEncoder(config)
        self.structure_encoder = StructureEncoder(config=config,
                                                  label_map=vocab.v2i['label'],
                                                  device=self.device,
                                                  graph_model_type=config.structure_encoder.type)

        self.propagation = Propagation(config=config,
                                 device=self.device,
                                 graph_model=self.structure_encoder,
                                 label_map=self.label_map)

    def optimize_params_dict(self):
        params = list()
        params.append({'params': self.text_encoder.parameters()})
        params.append({'params': self.token_embedding.parameters()})
        params.append({'params': self.hiagm.parameters()})
        return params

    def forward(self, batch):

        embedding,token_output = self.token_embedding(batch['token'].to(self.config.train.device_setting.device))
        logits = self.propagation(token_output)

        return logits
