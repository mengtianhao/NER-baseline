import torch
import torch.nn as nn
from torchcrf import CRF


class MyCRF(nn.Module):
    def __init__(self, labels_num):
        super(MyCRF, self).__init__()
        self.label_size = labels_num
        self.crf = CRF(self.label_size, batch_first=True)

    def forward(self, sentence, mask, labels=None, is_test=False):
        if not is_test:  # Training，return loss
            loss = -self.crf.forward(sentence, labels, mask, reduction='mean')
            decode = self.crf.decode(sentence, mask)
            return loss, decode
        else:  # Testing，return decoding
            decode = self.crf.decode(sentence, mask)
            return decode
