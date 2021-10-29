import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModel

class PretrainedTransformerEncoder(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.name = name
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).to(self.device)
        self.output_dim = self.get_output_dim()

    def get_output_dim(self):
        doc = 'foo bar'
        return self.model(doc).shape[0]

    def forward(self, doc):
        src = self.tokenizer(doc, return_tensors='torch')
        src = src.to(self.device)
        with torch.no_grad():
            tgt = self.model(**src)
        return tgt

    def encode(self, doc, return_tensors='np', agg='mean'):
        encoding = self.model(doc)
        if self.device == 'cuda:0':
            encoding = encoding.cpu()
        if agg == 'mean':
            encoding = encoding.mean(dim=1)
        if agg == 'max':
            encoding = encoding.max(dim=1)
        if agg == 'min':
            encoding = encoding.min(dim=1)
        if agg == 'median':
            encoding = torch.median(encoding, dim=1).values
        if return_tensors == 'torch':
            return encoding
        elif return_tensors == 'np':
            return encoding.numpy()
        else:
            raise AttributeError(f'Unknown return type: {return_tensors}')
