# -*- coding: utf-8 -*-

# @Author  : xmh
# @Time    : 2021/3/11 16:38
# @File    : trainer_rel.py

"""
file description:：

"""

# coding=utf-8
import sys
sys.path.append('/home/xieminghui/Projects/EntityRelationExtraction/')


import torch
import torch.nn as nn
from tqdm import tqdm
from utils.config_rel import ConfigRel, USE_CUDA
from modules.model_rel import AttBiLSTM
from data_loader.process_rel import DataPreparationRel
import numpy as np
import codecs


class Trainer:
    def __init__(self,
                 model,
                 config,
                 train_dataset=None,
                 dev_dataset=None,
                 test_dataset=None
                 ):
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset
        
        if USE_CUDA:
            self.model = self.model.cuda()
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.5,
                                                                    patience=8, min_lr=1e-5, verbose=True)
        self.get_id2rel()
        
    def train(self):
        print('STARTING TRAIN...')
        self.num_sample_total = len(self.train_dataset) * self.config.batch_size
        loss_eval_best = 1e8
        for epoch in range(self.config.epochs):
            print("Epoch: {}".format(epoch))
            pbar = tqdm(enumerate(self.train_dataset), total=len(self.train_dataset))
            loss_rel_total = 0.0
            self.optimizer.zero_grad()
            for i, data_item in pbar:
                loss_rel, pred_rel = self.model(data_item)
                loss_rel.backward()
                self.optimizer.step()

                loss_rel_total += loss_rel

            print("train rel loss: {0}".format(loss_rel_total / self.num_sample_total))
            
            if (epoch + 1) % 1 == 0:
                loss_rel_ave = self.evaluate()
                
            if epoch > 0 and (epoch+1) % 2 == 0:
                if loss_rel_ave < loss_eval_best:
                    loss_eval_best = loss_rel_ave
                    torch.save({
                        'epoch': epoch + 1, 'state_dict': self.model.state_dict(), 'loss_rel_best': loss_eval_best,
                        'optimizer': self.optimizer.state_dict(),
                    },
                        self.config.ner_checkpoint_path + str(epoch) + 'm-' + 'loss' +
                        str("%.2f" % loss_rel_ave) + 'ccks2019_rel.pth'
                    )
    
    def evaluate(self):
        print('STARTING EVALUATION...')
        self.model.train(False)
        pbar_dev = tqdm(enumerate(self.dev_dataset), total=len(self.dev_dataset))
    
        loss_rel_total = 0
        for i, data_item in pbar_dev:
            loss_rel, pred_rel = self.model(data_item)
            loss_rel_total += loss_rel
        
        self.model.train(True)
        loss_rel_ave = loss_rel_total / (len(self.dev_dataset) * self.config.batch_size)
        print("eval rel loss: {0}".format(loss_rel_ave))
        
        print(data_item['text'][1])
        print("subject: {0}, object：{1}".format(data_item['subject'][1], data_item['object'][1]))
        print("object rel: {}".format(self.id2rel[int(data_item['relation'][1])]))
        print("predict rel: {}".format(self.id2rel[int(pred_rel[1])]))
        return loss_rel_ave
    
    def get_id2rel(self):
        self.id2rel = {}
        for i, rel in enumerate(self.config.relations):
            self.id2rel[i] = rel

    def predict(self):
        print('STARTING PREDICTING...')
        self.model.train(False)
        pbar = tqdm(enumerate(self.test_dataset), total=len(self.test_dataset))
        for i, data_item in pbar:
            pred_rel = self.model(data_item, is_test=True)
        self.model.train(True)
        rel_pred = [[] for _ in range(len(pred_rel))]
        for i in range(len(pred_rel)):
            # for item in pred_rel[i]:
            rel_pred[i].append(self.id2rel[int(pred_rel[i])])
        return rel_pred

def get_embedding_pre():
    # token2id = {}
    # with open('../data/vocab.txt', 'r', encoding='utf-8') as f:
    #     cnt = 0
    #     for line in f:
    #         line = line.rstrip().split()
    #         token2id[line[0]] = cnt
    #         cnt += 1

    word2id = {}
    with codecs.open('../data/vec.txt', 'r', encoding='utf-8') as f:
        cnt = 0
        for line in f.readlines():
            word2id[line.split()[0]] = cnt
            cnt += 1

    word2vec = {}
    with codecs.open('../data/vec.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            word2vec[line.split()[0]] = list(map(eval, line.split()[1:]))
        unkown_pre = []
        unkown_pre.extend([1]*100)
    embedding_pre = []
    embedding_pre.append(unkown_pre)
    for word in word2id:
        if word in word2vec:
            embedding_pre.append(word2vec[word])
        else:
            embedding_pre.append(unkown_pre)
    embedding_pre = np.array(embedding_pre)
    return embedding_pre



if __name__ == '__main__':
    # PATH_NER = '../models/sequence_labeling/60m-f589.90n40236.67ccks2019_ner.pth'
    # ner_model_dict = torch.load(PATH_NER)

    print("Run EntityRelationExtraction REL ...")
    config = ConfigRel()
    embedding_pre = get_embedding_pre()
    # embedding_pre = ner_model_dict['state_dict']['word_embedding.weight']
    model = AttBiLSTM(config, embedding_pre)
    data_processor = DataPreparationRel(config)
    train_loader, dev_loader, test_loader = data_processor.get_train_dev_data(
        '../data/train_small.json',
    '../data/dev_small.json',
    '../data/predict.json')
    # train_loader, dev_loader, test_loader = data_processor.get_train_dev_data('../data/train_data_small.json')
    trainer = Trainer(model, config, train_loader, dev_loader, test_loader)
    trainer.train()