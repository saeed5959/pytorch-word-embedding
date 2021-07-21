import torch
import numpy as np
from torch import nn


class embedding(nn.Module):
    def __init__(self,voc_size,embed_size):
        super(embedding, self).__init__()
        self.voc_size = voc_size
        self.embed_size = embed_size
        self.embed_layer = nn.Embedding(self.voc_size,self.embed_size)

    def forward(self,input):
        input_embed = self.embed_layer(input)
        return input_embed


model = embedding(3,2)
out = model(torch.tensor([1]))
print(out)