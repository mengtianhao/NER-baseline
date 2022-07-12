import torch
import torch.nn as nn
from torch_utils.bi_lstm import BiLSTM


class MyModel(nn.Module):
    def __init__(self, config, pretrained_class, pretrained_path, mid_struct,
                 num_labels):
        super(MyModel, self).__init__()
        # 预训练模型
        print(pretrained_path)
        self.encoder = pretrained_class.from_pretrained(pretrained_path)
        self.config = config
        self.mid_struct = mid_struct
        self.num_labels = num_labels
        # 结构
        if self.mid_struct == 'bilstm':
            self.mid_struct_model = BiLSTM(self.num_labels, embedding_size=config.hidden_size,
                                           hidden_size=config.lstm_hidden, num_layers=config.num_layers,
                                           dropout=config.drop_prob, with_ln=True)
        #
        self.base_output = nn.Linear(config.hidden_size, self.num_labels).to(config.device)

    def forward(self, input_ids, token_type_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, token_type_ids=token_type_ids,
                               attention_mask=attention_mask)
        # print(outputs[0].shape)     # (batch_size, seq_len, tag_size)
        if self.mid_struct == 'base':
            outputs = self.base_output(outputs[0])
        elif self.mid_struct == 'bilstm':
            """
            :param embed: (seq_len, batch_size, embedding_size)
            :param mask: (seq_len, batch_size)
            :return lstm_features: (seq_len, batch_size, tag_size)
            """
            outputs = self.mid_struct_model.get_lstm_features(outputs[0].transpose(1, 0), attention_mask.transpose(1, 0)).transpose(1, 0)

        return outputs
