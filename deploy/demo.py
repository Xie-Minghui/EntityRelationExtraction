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

def test():
    test_path = './test.json'
    PATH_NER = '../models/sequence_labeling/60m-f589.90n40236.67ccks2019_ner.pth'
    config_ner = ConfigNer()
    ner_model = SeqLabel(config_ner)
    ner_model_dict = torch.load(PATH_NER)
    ner_model.load_state_dict(ner_model_dict)
    ConfigNer.batch_size = 1
    ner_data_process = ModelDataPreparation(ConfigNer)
    _, _, test_loader = ner_data_process.get_train_dev_data(path_test=test_path)
    trainerNer = trainer_ner.Trainer(ner_model, config_ner, test_dataset=test_loader)
    pred_ner = trainerNer.predict()
    print("haha")
    print(pred_ner)
    

if __name__ == '__main__':
    test()
