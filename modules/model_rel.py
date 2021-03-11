# -*- coding: utf-8 -*-

# @Author  : xmh
# @Time    : 2021/3/11 16:36
# @File    : model_rel.py

"""
file description:：

"""
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)  # 使用相同的初始化种子，保证每次初始化结果一直，便于调试


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.query = nn.Parameter(torch.randn(config.batch_size, 1, config.hidden_dim_lstm*2))  # [batch, 1, hidden_dim]
    
    def forward(self, H):
        M = torch.tanh(H)  # H [batch_size, sentence_length, hidden_dim_lstm]
        attention_prob = torch.matmul(M, self.query.transpose(-1, -2))  # [batch_size, sentence_length, 1]
        alpha = F.softmax(attention_prob,dim=-1)
        attention_output = torch.matmul(alpha.transpose(-1, -2), H)  # [batch_size, 1, hidden_dim_lstm]
        attention_output = attention_output.squeeze()
        attention_output = torch.tanh(attention_output)
        return attention_output


class AttBiLSTM(nn.Module):
    def __init__(self, config, embedding_pre=None):
        super().__init__()
        self.embedding_dim = config.embedding_dim
        self.vocab_size = config.vocab_size
        self.hidden_dim = config.hidden_dim_lstm
        self.num_layers = config.num_layers
        self.batch_size = config.batch_size
        self.dropout = nn.Dropout(config.dropout_embedding)
        self.pretrained = config.pretrained
        self.config = config
        
        assert (self.pretrained is True and embedding_pre is not None) or \
               (self.pretrained is False and embedding_pre is None), "预训练必须有训练好的embedding_pre"
        # 定义网络层
        # 对于关系抽取，命名实体识别和关系抽取共享编码层
        if self.pretrained:
            self.word_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_pre), freeze=False)
        else:
            self.word_embedding = nn.Embedding(config.vocab_size, config.embedding_dim, padding_idx=config.pad_token_id)
        
        # self.pos1_embedding = nn.Embedding(config.pos_size, config.embedding_dim)
        # self.pos2_embedding = nn.Embedding(config.pos_size, config.embedding_dim)
        self.gru = nn.GRU(config.embedding_dim, config.hidden_dim_lstm, num_layers=config.num_layers, batch_first=True, bidirectional=True,
                          dropout=config.dropout_lstm)
        self.attention_layer = Attention(config)
        self.classifier = nn.Linear(config.hidden_dim_lstm, config.num_relations)
    
    def forward(self, data_item, is_test=False):
        # embeddings = torch.cat((self.word_embedding(data_item['sentences']),
        #                         self.pos1_embedding(data_item['positionE1']),
        #                         self.pos2_embedding(data_item['positionE2'])),
        #                        1)  # [batch_size, sentence_length, embedding_dim]
        embeddings = self.word_embedding(data_item['sentence_cls'].to(torch.int64))
        if self.config.use_dropout:
            embeddings = self.dropout(embeddings)
        hidden_init = torch.randn(2 * self.num_layers, self.batch_size, self.hidden_dim)
        output, h_n = self.gru(embeddings, hidden_init)
        attention_output = self.attention_layer(output)
        # hidden_cls = torch.tanh(attention_output)
        if not is_test:
            loss = F.cross_entropy(attention_output, data_item['relation'])  # loss = F.cross_entropy(attention_output, data_item['relation'])

        pred = attention_output.argmax(dim=-1)
        if is_test:
            return pred
        return loss, pred
