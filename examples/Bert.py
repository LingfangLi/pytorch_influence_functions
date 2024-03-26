#! /usr/bin/env python3
from datasets import load_dataset, Dataset, DatasetDict
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformer import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(src_length, trg_length):
    # config = BertConfig.from_pretrained(PRETRAINED_LM)
    # config.num_labels = 6o
    # model = BertForSequenceClassification(config)
    model = Transformer(src_length, trg_length).to(device)
    sta_dic = torch.load(f'{MODEL_DIR}\model-epoch100.pt')
    print(type(sta_dic))
    model.load_state_dict(sta_dic, strict=False)
    model = model.to(device)
    return model
