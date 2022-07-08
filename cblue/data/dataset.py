import numpy as np
import torch
from torch.utils.data import Dataset
from cblue.models import convert_examples_to_features_for_tokens


class EEDataset(Dataset):
    def __init__(
            self,
            samples,
            data_processor,
            tokenizer,
            mode='train',
            max_length=128,
            ignore_label=-100,
            model_type='bert',
            ngram_dict=None
    ):
        super(EEDataset, self).__init__()

        self.orig_text = samples['orig_text']
        self.texts = samples['text']
        if mode != "test":
            self.labels = samples['label']
        else:
            self.labels = None

        self.data_processor = data_processor
        self.tokenizer = tokenizer
        self.ignore_label = ignore_label
        self.max_length = max_length
        self.mode = mode
        self.ngram_dict = ngram_dict
        self.model_type = model_type

    def __getitem__(self, idx):
        text = self.texts[idx]
        if self.model_type == 'zen':
            inputs = convert_examples_to_features_for_tokens(text, max_seq_length=self.max_length,
                                                             ngram_dict=self.ngram_dict, tokenizer=self.tokenizer,
                                                             return_tensors=True)
        else:
            inputs = self.tokenizer.encode_plus(text, max_length=self.max_length, padding='max_length', truncation=True)
        if self.mode != "test":
            label = [self.data_processor.label2id[label_] for label_ in
                     self.labels[idx].split('\002')]  # find index from label list
            label = ([-100] + label[:self.max_length - 2] + [-100] +
                     [self.ignore_label] * self.max_length)[:self.max_length]  # use ignore_label padding CLS+label+SEP
            if self.model_type == 'zen':
                return inputs['input_ids'], inputs['token_type_ids'], \
                       inputs['attention_mask'], torch.tensor(label), inputs['input_ngram_ids'], \
                       inputs['ngram_attention_mask'], inputs['ngram_token_type_ids'], \
                       inputs['ngram_position_matrix']
            else:
                return np.array(inputs['input_ids']), np.array(inputs['token_type_ids']), \
                    np.array(inputs['attention_mask']), np.array(label)
        else:
            if self.model_type == 'zen':
                return inputs['input_ids'], inputs['token_type_ids'], \
                       inputs['attention_mask'], inputs['input_ngram_ids'], \
                       inputs['ngram_attention_mask'], inputs['ngram_token_type_ids'], \
                       inputs['ngram_position_matrix']
            else:
                return np.array(inputs['input_ids']), np.array(inputs['token_type_ids']), \
                       np.array(inputs['attention_mask']),

    def __len__(self):
        return len(self.texts)

















