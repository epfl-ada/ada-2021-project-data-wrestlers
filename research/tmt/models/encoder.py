import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoConfig
import math
from tqdm import tqdm

class PretrainedTransformerEncoder(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.name = name
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_config(
            AutoConfig.from_pretrained(name)
        ).to(self.device)
        self.output_dim = self.get_output_dim()

    def get_output_dim(self):
        doc = 'foo bar'
        embedding = self(doc)
        return embedding.size(-1)

    def forward(self, doc, agg='mean', batch_size=None):
        src = self.tokenizer(doc, return_tensors='pt', padding=True, truncation=True)
        src = src.to(self.device)
        batch_size = len(doc) if batch_size is None else batch_size
        encodings = []
        for n in tqdm(range(math.ceil(len(doc)/batch_size))):
            with torch.no_grad():
                tmp = self.model(**{key: val[n*batch_size:(n+1)*batch_size] for key, val in src.items()}).last_hidden_state
                encodings.append(tmp)
        if self.device == 'cuda:0':
            encoding = encoding.cpu()
        if agg == 'mean':
            encodings = [encoding.mean(dim=1) for encoding in encodings]
        if agg == 'max':
            encodings = [encoding.max(dim=1) for encoding in encodings]
        if agg == 'min':
            encodings = [encoding.min(dim=1) for encoding in encodings]
        if agg == 'median':
            encodings = [torch.median(encoding, dim=1).values for encoding in encodings]
        encodings = torch.cat(encodings, dim=0)
        # with torch.no_grad():
        #     tgt = self.model(**src).last_hidden_state
        return encodings

    def encode(self, doc, batch_size=None, return_tensors='np', agg='mean'):
        # encodings = []
        # for n in tqdm(range(math.ceil(len(doc)/batch_size))):
        #     tmp = self(doc[n*batch_size:(n+1)*batch_size])
        #     encodings.append(tmp)
        # if self.device == 'cuda:0':
        #     encoding = encoding.cpu()
        # if agg == 'mean':
        #     encodings = [encoding.mean(dim=1) for encoding in encodings]
        # if agg == 'max':
        #     encodings = [encoding.max(dim=1) for encoding in encodings]
        # if agg == 'min':
        #     encodings = [encoding.min(dim=1) for encoding in encodings]
        # if agg == 'median':
        #     encodings = [torch.median(encoding, dim=1).values for encoding in encodings]
        # encodings = torch.cat(encodings, dim=0)
        encodings = self(doc=doc, agg=agg, batch_size=batch_size)
        if return_tensors == 'torch':
            return encodings
        elif return_tensors == 'np':
            return encodings.numpy()
        else:
            raise AttributeError(f'Unknown return type: {return_tensors}')
