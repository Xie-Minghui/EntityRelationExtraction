# -*- coding: utf-8 -*-

# @Author  : xmh
# @Time    : 2021/3/3 17:28
# @File    : trainer_ner.py

"""
file description:：

"""
import sys
sys.path.append('/home/xieminghui/Projects/EntityRelationExtraction/')  # 添加路径

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from utils.config_ner import ConfigNer, USE_CUDA
from modules.model_ner import SeqLabel
from data_loader.process_ner import ModelDataPreparation
import math

from seqeval.metrics import f1_score
from seqeval.metrics import precision_score
from seqeval.metrics import accuracy_score
from seqeval.metrics import recall_score
from seqeval.metrics import classification_report

class Trainer:
    def __init__(self,
                 model,
                 config,
                 train_dataset=None,
                 dev_dataset=None,
                 test_dataset=None,
                 ):
        self.model = model
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset
        self.config = config
        
        if USE_CUDA:
            self.model = self.model.cuda()
        # 初始优化器
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)
        # 学习率调控
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.5,
                                                                   patience=8, min_lr=1e-5, verbose=True)
        self.get_id2token_type()

    def get_id2token_type(self):
        self.id2token_type = {}
        for i, token_type in enumerate(self.config.token_types):
            self.id2token_type[i] = token_type
        
    def train(self):
        print('STARTING TRAIN...')
        f1_ner_total_best = 0.0
        self.num_sample_total = len(self.train_dataset) * self.config.batch_size
        for epoch in range(self.config.epochs):
            print("Epoch: {}".format(epoch))
            pbar = tqdm(enumerate(self.train_dataset), total=len(self.train_dataset))
            loss_ner_total, f1_ner_total = 0, 0
            for i, data_item in pbar:
                loss_ner, f1_ner, pred_ner = self.train_batch(data_item)
                loss_ner_total += loss_ner
                f1_ner_total += f1_ner
                
            if (epoch+1) % 1 == 0:
                self.predict_sample()
            print("train ner loss: {0}, f1 score: {1}".format(loss_ner_total/self.num_sample_total,
                                                        f1_ner_total/self.num_sample_total*self.config.batch_size))
            # pbar.set_description('TRAIN LOSS: {}'.format(loss_total/self.num_sample_total))
            if (epoch+1) % 1 == 0:
                self.evaluate()
            if epoch > 8 and f1_ner_total >= f1_ner_total_best:
                f1_ner_total_best = f1_ner_total
                torch.save({
                    'epoch': epoch+1, 'state_dict': self.model.state_dict(), 'f1_best': f1_ner_total,
                    'optimizer': self.optimizer.state_dict(),
                },
                self.config.ner_checkpoint_path + str(epoch) + 'm-' + 'f'+str("%.2f"%f1_ner_total) + 'n'+
                    str("%.2f"%loss_ner_total) +'ccks2019_ner.pth'
                )
    
    def train_batch(self, data_item):
        self.optimizer.zero_grad()
        loss_ner, pred_ner = self.model(data_item)
        pred_token_type = self.restore_ner(pred_ner, data_item['mask_tokens'])
        f1_ner = f1_score(data_item['token_type_origin'], pred_token_type)
        loss_ner.backward()
        self.optimizer.step()
        
        return loss_ner,f1_ner, pred_ner
    
    def restore_ner(self, pred_ner, mask_tokens):
        pred_token_type = []
        for i in range(len(pred_ner)):
            list_tmp = []
            for j in range(len(pred_ner[0])):
                if mask_tokens[i, j] == 0:
                    break
                list_tmp.append(self.id2token_type[pred_ner[i][j]])
            pred_token_type.append(list_tmp)
            
        return pred_token_type
    
    def evaluate(self):
        print('STARTING EVALUATION...')
        self.model.train(False)
        pbar_dev = tqdm(enumerate(self.dev_dataset), total=len(self.dev_dataset))
        
        loss_total, loss_ner_total = 0, 0
        for i, data_item in pbar_dev:
            loss_ner, pred_ner = self.model(data_item)
            loss_ner_total += loss_ner
        
        self.model.train(True)
        print("eval ner loss: {0}".format(loss_ner_total / (len(self.dev_dataset) * self.config.batch_size)))
        # return loss_ner_total / (len(self.dev_dataset) * self.config.batch_size)
    
    def predict(self):
        print('STARTING PREDICTING...')
        self.model.train(False)
        pbar = tqdm(enumerate(self.test_dataset), total=len(self.test_dataset))
        for i, data_item in pbar:
            pred_ner = self.model(data_item, is_test=True)
        self.model.train(True)
        token_pred = [[] for _ in range(len(pred_ner))]
        for i in range(len(pred_ner)):
            for item in pred_ner[i]:
                token_pred[i].append(self.id2token_type[item])
        return token_pred

    def predict_sample(self):
        print('STARTING TESTING...')
        self.model.train(False)
        pbar = tqdm(enumerate(self.test_dataset), total=len(self.test_dataset))

        for i, data_item in pbar:
            pred_ner = self.model(data_item, is_test=True)
        data_item0 = data_item
        pred_ner = pred_ner[0]
        token_pred = []
        for i in pred_ner:
            token_pred.append(self.id2token_type[i])
        print("token_pred: {}".format(token_pred))
        print(data_item0['text'][0])
        print(data_item0['spo_list'][0])
        self.model.train(True)
        
        
if __name__ == '__main__':
    print("Run EntityRelationExtraction NER ...")
    config = ConfigNer()
    model = SeqLabel(config)
    data_processor = ModelDataPreparation(config)
    train_loader, dev_loader, test_loader = data_processor.get_train_dev_data(
        '../data/train_data_small.json',
    '../data/dev_small.json',
    '../data/predict.json')
    # train_loader, dev_loader, test_loader = data_processor.get_train_dev_data('../data/train_data_small.json')
    trainer = Trainer(model, config, train_loader, dev_loader, test_loader)
    trainer.train()