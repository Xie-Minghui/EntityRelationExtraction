# -*- coding: utf-8 -*-

# @Author  : xmh
# @Time    : 2021/3/11 19:34
# @File    : demo.py

"""
file description:

"""
import torch
from modules.model_ner import SeqLabel
from modules.model_rel import AttBiLSTM
from utils.config_ner import ConfigNer, USE_CUDA
from utils.config_rel import ConfigRel

from data_loader.process_ner import ModelDataPreparation
from data_loader.process_rel import DataPreparationRel

from mains import trainer_ner, trainer_rel
import json
from transformers import BertForSequenceClassification


def get_entities(pred_ner, text):
    token_types = [[] for _ in range(len(pred_ner))]
    entities = [[] for _ in range(len(pred_ner))]
    for i in range(len(pred_ner)):
        token_type = []
        entity = []
        j = 0
        word_begin = False
        while j < len(pred_ner[i]):
            if pred_ner[i][j][0] == 'B':
                if word_begin:
                    token_type = []  # 防止多个B出现在一起
                    entity = []
                token_type.append(pred_ner[i][j])
                entity.append(text[i][j])
                word_begin = True
            elif pred_ner[i][j][0] == 'I':
                if word_begin:
                    token_type.append(pred_ner[i][j])
                    entity.append(text[i][j])
            else:
                if word_begin:
                    token_types[i].append(''.join(token_type))
                    token_type = []
                    entities[i].append(''.join(entity))
                    entity = []
                word_begin = False
            j += 1
    return token_types, entities


def test():
    # test_path = './test.json'
    # PATH_NER = '../models/sequence_labeling/60m-f589.90n40236.67ccks2019_ner.pth'
    # config_ner = ConfigNer()
    # # config_ner.batch_size = 1
    # ner_model = SeqLabel(config_ner)
    # ner_model_dict = torch.load(PATH_NER)
    # ner_model.load_state_dict(ner_model_dict['state_dict'])
    #
    # ner_data_process = ModelDataPreparation(config_ner)
    # _, _, test_loader = ner_data_process.get_train_dev_data(path_test=test_path)
    # trainerNer = trainer_ner.Trainer(ner_model, config_ner, test_dataset=test_loader)
    # pred_ner = trainerNer.predict()
    # # print("haha")
    # print(pred_ner)
    # text = None
    # for data_item in test_loader:
    #     text = data_item['text']
    # token_types, entities = get_entities(pred_ner, text)
    # print(token_types)
    # print(entities)
    #
    # rel_list = []
    # with open('./rel_predict.json', 'w', encoding='utf-8') as f:
    #     for i in range(len(pred_ner)):
    #         texti = text[i]
    #         for j in range(len(entities[i])):
    #             for k in range(len(entities[i])):
    #                 if j == k:
    #                     continue
    #                 rel_list.append({"text":texti, "spo_list":{"subject": entities[i][j], "object": entities[i][k]}})
    #     json.dump(rel_list, f, ensure_ascii=False)

    PATH_REL = '../models/rel_cls/25m-acc0.93ccks2019_rel.pth'

    config_rel = ConfigRel()
    config_rel.batch_size = 8
    rel_model = BertForSequenceClassification.from_pretrained('../bert-base-chinese', num_labels=config_rel.num_relations)
    # rel_model = AttBiLSTM(config_rel)
    rel_model_dict = torch.load(PATH_REL)
    rel_model.load_state_dict(rel_model_dict['state_dict'])
    rel_test_path = './rel_predict.json'

    rel_data_process = DataPreparationRel(config_rel)
    _, _, test_loader = rel_data_process.get_train_dev_data(path_test=rel_test_path)
    trainREL = trainer_rel.Trainer(rel_model, config_rel, test_dataset=test_loader)
    rel_pred = trainREL.bert_predict()
    print(rel_pred)





if __name__ == '__main__':
    test()
