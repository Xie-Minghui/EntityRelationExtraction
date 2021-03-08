# -*- coding: utf-8 -*-

# @Author  : xmh
# @Time    : 2021/3/3 10:02
# @File    : joint_model.py

"""
file description:：

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF
import numpy as np
from utils.config import USE_CUDA

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class JointModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        setup_seed(1)
        
        self.vocab_size = config.vocab_size
        self.embedding_dim = config.embedding_dim
        self.hidden_dim = config.hidden_dim_lstm
        self.num_layers = config.num_layers
        self.batch_size = config.batch_size
        self.layer_size = config.layer_size  # self.hidden_dim, 之前这里没有改
        self.num_token_type = config.num_token_type  # 实体类型的综述
        self.config = config
        
        self.word_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.token_type_embedding = nn.Embedding(config.num_token_type, config.token_type_dim)
        self.gru = nn.GRU(config.embedding_dim, config.hidden_dim_lstm, num_layers=config.num_layers, batch_first=True,
                          bidirectional=True)
        self.is_train = True
        if USE_CUDA:
            self.weights_rel = (torch.ones(self.config.num_relations) * 100).cuda()
        else:
            self.weights_rel = torch.ones(self.config.num_relations) * 100
        self.weights_rel[0] = 1

        self.V_ner = nn.Parameter(torch.rand((config.num_token_type, self.layer_size)))
        self.U_ner = nn.Parameter(torch.rand((self.layer_size, 2 * self.hidden_dim)))
        self.b_s_ner = nn.Parameter(torch.rand(self.layer_size))
        self.b_c_ner = nn.Parameter(torch.rand(config.num_token_type))

        self.U_head = nn.Parameter(torch.rand((self.layer_size, self.hidden_dim * 2 + self.config.token_type_dim)))
        self.W_head = nn.Parameter(torch.rand((self.layer_size, self.hidden_dim * 2 + self.config.token_type_dim)))
        self.V_head = nn.Parameter(torch.rand(self.layer_size, len(self.config.relations)))
        self.b_s_head = nn.Parameter(torch.rand(self.layer_size))
        
        self.dropout_embedding_layer = torch.nn.Dropout(config.dropout_embedding)
        self.dropout_head_layer = torch.nn.Dropout(config.dropout_head)
        self.dropout_ner_layer = torch.nn.Dropout(config.dropout_ner)
        self.dropout_lstm_layer = torch.nn.Dropout(config.dropout_lstm)
        self.crf_model = CRF(self.num_token_type, batch_first=True)
        
    def get_ner_score(self, output_lstm):
        
        res = torch.matmul(output_lstm, self.U_ner.transpose(-1, -2)) + self.b_s_ner # [seq_len, batch, self.layer_size]
        res = torch.tanh(res)
        # res = F.leaky_relu(res,  negative_slope=0.01)
        if self.config.use_dropout:
            res = self.dropout_ner_layer(res)
            
        ans = torch.matmul(res, self.V_ner.transpose(-1, -2)) + self.b_c_ner  # [seq_len, batch, num_token_type]
        
        return ans
    
    def broadcasting(self, left, right):
        left = left.permute(1, 0, 2)
        left = left.unsqueeze(3)
        
        right = right.permute(0, 2, 1)
        right = right.unsqueeze(0)
        
        B = left + right  # [seq_len, batch, layer_size, seq_len] = [seq_len, batch, layer_size, 1] + [1, batch, layer_size, seq_len]
        B = B.permute(1, 0, 3, 2)
        
        return B  # [batch, seq_len, seq_len, layer_size]
    
    def getHeadSelectionScores(self, rel_input):
       
        left = torch.matmul(rel_input, self.U_head.transpose(-1, -2))  # [batch, seq, self.layer_size]
        right = torch.matmul(rel_input, self.W_head.transpose(-1, -2))
        
        out_sum = self.broadcasting(left, right)
        out_sum_bias = out_sum + self.b_s_head
        # out_sum_bias = torch.tanh(out_sum_bias)  # relu
        out_sum_bias = F.elu(out_sum_bias)
        # out_sum_bias = F.leaky_relu(out_sum_bias,  negative_slope=0.01)  # relu
        if self.config.use_dropout:
            out_sum_bias = self.dropout_head_layer(out_sum_bias)
        res = torch.matmul(out_sum_bias, self.V_head)  # [layer_size, num_relation] [batch,..., num_relation]
        # res的维度应该是 [batch, seq_len, seq_len, num_relation]
        # if self.config.use_dropout:
        #     res = self.dropout_head_layer(res)
        return res
    
    def forward(self, data_item, is_test=False, hidden_init=None):
        # 因为不是多跳机制，所以hidden_init不能继承之前的最终隐含态
        '''
        
        :param data_item: data_item = {'',}
        :type data_item: dict
        :return:
        :rtype:
        '''
        # print("hello5")
        embeddings = self.word_embedding(data_item['text_tokened'].to(torch.int64))  # 要转化为int64
        if self.config.use_dropout:
            embeddings = self.dropout_embedding_layer(embeddings)
        # if hidden_init is None:
        # print("hello6")
        if USE_CUDA:
            hidden_init = torch.randn(2*self.num_layers, self.batch_size, self.hidden_dim).cuda()
        else:
            hidden_init = torch.randn(2 * self.num_layers, self.batch_size, self.hidden_dim)
        output_lstm, h_n =self.gru(embeddings, hidden_init)
        # output_lstm [batch, seq_len, 2*hidden_dim]  h_n [2*num_layers, batch, hidden_dim]
        # print("hello7")
        # if self.config.use_dropout:
        #     output_lstm = self.dropout_lstm_layer(output_lstm)  # 用了效果变差
        ner_score = self.get_ner_score(output_lstm)
        # print("hello0")
        # 下面是使用CFR
        
        if USE_CUDA:
            self.crf_model = self.crf_model.cuda()
        if not is_test:
            log_likelihood = self.crf_model(ner_score, data_item['token_type_list'].to(torch.int64),
                                       mask=data_item['mask_tokens'])
            loss_ner = -log_likelihood
            
        pred_ner = self.crf_model.decode(ner_score)  # , mask=data_item['mask_tokens']
        
        # 下面使用的是Softmax
        # loss_ner = F.softmax(ner_score, data_item['ner_type'])
        # pred_ner = torch.argmax(ner_score, 2)
        
        #--------------------------Relation
        if not is_test and torch.rand(1) > self.config.teach_rate:
            labels = data_item['token_type_list']
        else:
            if USE_CUDA:
                labels = torch.Tensor(pred_ner).cuda()
            else:
                labels = torch.Tensor(pred_ner)
        # print("hello1")
        label_embeddings = self.token_type_embedding(labels.to(torch.int64))
        rel_input = torch.cat((output_lstm, label_embeddings), 2)
        rel_score_matrix = self.getHeadSelectionScores(rel_input)  # [batch, seq_len, seq_len, num_relation]
        rel_score_prob = torch.sigmoid(rel_score_matrix)
        #gold_predicate_matrix_one_hot = F.one_hot(data_item['pred_rel_matrix'], len(self.config.relations))
        if not is_test:
            # 这样计算交叉熵有问题吗
            # 交叉熵计算不适用 rel_score_prob， 应该是rel_score_matrix
            loss_rel = F.cross_entropy(rel_score_prob.permute(0, 3, 1, 2), data_item['pred_rel_matrix'], self.weights_rel)  # 要把分类放在第二维度
            loss_rel *= rel_score_prob.shape[1]
        rel_score_prob = rel_score_prob - (self.config.threshold_rel - 0.5)  # 超过了一定阈值之后才能判断关系
        pred_rel = torch.round(rel_score_prob).to(torch.int64)
        # print("hello2")
        if is_test:
            return pred_ner, pred_rel
        # loss_ner = min(loss_ner, 30000)
        # loss_rel = min(loss_rel, 10)
        return loss_ner, loss_rel, pred_ner, pred_rel


