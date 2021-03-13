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
from utils.config_rel import USE_CUDA

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)   # 使用相同的初始化种子，保证每次初始化结果一直，便于调试


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        setup_seed(1)
        # self.query = nn.Parameter(torch.randn(1, config.hidden_dim_lstm))  # [batch, 1, hidden_dim]
    
    # def forward(self, H):
    #     M = torch.tanh(H)  # H [batch_size, sentence_length, hidden_dim_lstm]
    #     attention_prob = torch.matmul(M, self.query.transpose(-1, -2))  # [batch_size, sentence_length, 1]
    #     alpha = F.softmax(attention_prob,dim=-1)
    #     attention_output = torch.matmul(alpha.transpose(-1, -2), H)  # [batch_size, 1, hidden_dim_lstm]
    #     attention_output = attention_output.squeeze(axis=1)
    #     attention_output = torch.tanh(attention_output)
    #     return attention_output
    
    def forward(self, output_lstm, hidden_lstm):
        hidden_lstm = torch.sum(hidden_lstm, dim=0)
        att_weights = torch.matmul(output_lstm, hidden_lstm.unsqueeze(2)).squeeze(2)
        alpha = F.softmax(att_weights, dim=1)
        new_hidden = torch.matmul(output_lstm.transpose(-1, -2), alpha.unsqueeze(2)).squeeze(2)
        
        return new_hidden

class AttBiLSTM(nn.Module):
    def __init__(self, config, embedding_pre=None):
        super().__init__()
        setup_seed(1)
        self.embedding_dim = config.embedding_dim
        self.vocab_size = config.vocab_size
        self.hidden_dim = config.hidden_dim_lstm
        self.num_layers = config.num_layers
        self.batch_size = config.batch_size
        self.embed_dropout = nn.Dropout(config.dropout_embedding)
        self.lstm_dropout = nn.Dropout(config.dropout_lstm_output)
        self.pretrained = config.pretrained
        self.config = config
        self.relation_embed_layer = nn.Embedding(config.num_relations, self.hidden_dim)
        self.relations = torch.Tensor([i for i in range(config.num_relations)])
        if USE_CUDA:
            self.relations = self.relations.cuda()
        self.relation_bias = nn.Parameter(torch.randn(config.num_relations))
        
        assert (self.pretrained is True and embedding_pre is not None) or \
               (self.pretrained is False and embedding_pre is None), "预训练必须有训练好的embedding_pre"
        # 定义网络层
        # 对于关系抽取，命名实体识别和关系抽取共享编码层
        if self.pretrained:
            # self.word_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_pre), freeze=False)
            self.word_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_pre), freeze=False)
        else:
            self.word_embedding = nn.Embedding(config.vocab_size, config.embedding_dim, padding_idx=config.pad_token_id)
        
        # self.pos1_embedding = nn.Embedding(config.pos_size, config.embedding_dim)
        # self.pos2_embedding = nn.Embedding(config.pos_size, config.embedding_dim)
        self.gru = nn.GRU(config.embedding_dim+2*config.pos_dim, config.hidden_dim_lstm, num_layers=config.num_layers, batch_first=True, bidirectional=True,
                          dropout=config.dropout_lstm)
        self.attention_layer = Attention(config)
        # self.classifier = nn.Linear(config.hidden_dim_lstm, config.num_relations)

        if USE_CUDA:
            self.weights_rel = (torch.ones(self.config.num_relations) * 6).cuda()
        else:
            self.weights_rel = torch.ones(self.config.num_relations) * 6
        # self.weights_rel[9], self.weights_rel[13], self.weights_rel[14], self.weights_rel[46] = 100, 100, 100, 100
        self.weights_rel[0] = 1
        self.hidden_init = torch.randn(2 * self.num_layers, self.batch_size, self.hidden_dim)
        if USE_CUDA:
            self.hidden_init = self.hidden_init.cuda()
        # self.pos_embedding_layer = nn.Embedding(config.max_seq_length*4, config.pos_dim)
    
    def forward(self, data_item, is_test=False):

        word_embeddings = self.word_embedding(data_item['sentence_cls'].to(torch.int64))
        # pos1_embeddings = self.pos_embedding_layer(data_item['position_s'].to(torch.int64))
        # pos2_embeddings = self.pos_embedding_layer(data_item['position_o'].to(torch.int64))
        # embeddings = torch.cat((word_embeddings, pos1_embeddings, pos2_embeddings), 2)  # batch_size, seq, word_dim+2*pos_dim
        embeddings = word_embeddings
        if self.config.use_dropout:
            embeddings = self.embed_dropout(embeddings)

        output, h_n = self.gru(embeddings, self.hidden_init)
        if self.config.use_dropout:
            output = self.lstm_dropout(output)
        attention_input = output[:, :, :self.hidden_dim] + output[:, :, self.hidden_dim:]
        attention_output = self.attention_layer(attention_input, h_n)
        # hidden_cls = torch.tanh(attention_output)
        # output_cls = self.classifier(attention_output)
        relation_embeds = self.relation_embed_layer(self.relations.to(torch.int64))
        # res = torch.add(torch.matmul(attention_output, relation_embeds.transpose(-1, -2)), self.relation_bias)
        res = torch.matmul(attention_output, relation_embeds.transpose(-1, -2))

        if not is_test:
            loss = F.cross_entropy(res, data_item['relation'], self.weights_rel)  # loss = F.cross_entropy(attention_output, data_item['relation'])
            # loss /= self.config.batch_size
        res = F.softmax(res, -1)
        pred = res.argmax(dim=-1)
        if is_test:
            return pred
        return loss, pred
