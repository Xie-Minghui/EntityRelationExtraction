# -*- coding: utf-8 -*-

# @Author  : xmh
# @Time    : 2021/3/3 14:31
# @File    : data_process.py

"""
file description:：

"""

'''
针对spo_list的主客体在原文的token进行标注，第一个字标注B-type,后面的字标注I-type，文本中其他词标注为O
（先将所有文本标注为O，然后根据spo_list的内容，将对应位置覆盖）

'''
import json
import torch
import copy
from utils.config import Config, USE_CUDA
import numpy as np


class ModelDataPreparation:
    def __init__(self, config):
        self.config = config
        self.get_type_rel2id()
    
    def subject_object_labeling(self, spo_list, text_tokened):
        # 在列表 k 中确定列表 q 的位置
        def _index_q_list_in_k_list(q_list, k_list):
            """Known q_list in k_list, find index(first time) of q_list in k_list"""
            q_list_length = len(q_list)
            k_list_length = len(k_list)
            for idx in range(k_list_length - q_list_length + 1):
                t = [q == k for q, k in zip(q_list, k_list[idx: idx + q_list_length])]
                # print(idx, t)
                if all(t):
                    # print(idx)
                    idx_start = idx
                    return idx_start

        # 给主体和客体表上BIO分割式类型标签
        def _labeling_type(subject_object, so_type):
            so_tokened = [c for c in subject_object]
            so_tokened_length = len(so_tokened)
            idx_start = _index_q_list_in_k_list(q_list=so_tokened, k_list=text_tokened)
            if idx_start is None:
                tokener_error_flag = True
                '''
                实体: "1981年"  原句: "●1981年2月27日，中国人口学会成立"
                so_tokened ['1981', '年']  text_tokened ['●', '##19', '##81', '年', '2', '月', '27', '日', '，', '中', '国', '人', '口', '学', '会', '成', '立']
                so_tokened 无法在 text_tokened 找到！原因是bert_tokenizer.tokenize 分词增添 “##” 所致！
                '''
            else:  # 给实体开始处标 B 其它位置标 I
                labeling_list[idx_start] = "B-" + so_type
                if so_tokened_length == 2:
                    labeling_list[idx_start + 1] = "I-" + so_type
                elif so_tokened_length >= 3:
                    labeling_list[idx_start + 1: idx_start + so_tokened_length] = ["I-" + so_type] * (
                                so_tokened_length - 1)
            return idx_start

        labeling_list = ["O" for _ in range(len(text_tokened))]
        predicate_value_list = [[] for _ in range(len(text_tokened))]
        predicate_location_list = [[] for _ in range(len(text_tokened))]
        have_error = False
        for spo_item in spo_list:
            subject = spo_item["subject"]
            subject_type = spo_item["subject_type"]
            object = spo_item["object"]
            subject, object = map(self.get_rid_unkonwn_word, (subject, object))
            subject = list(map(lambda x: x.lower(), subject))
            object = list(map(lambda x: x.lower(), object))
            object_type = spo_item["object_type"]
            predicate_value = spo_item["predicate"]
            subject_idx_start = _labeling_type(subject, subject_type)
            object_idx_start = _labeling_type(object, object_type)
            if subject_idx_start is None or object_idx_start is None:
                have_error = True
                return labeling_list, predicate_value_list, predicate_location_list, have_error
            predicate_value_list[subject_idx_start].append(predicate_value)
            predicate_location_list[subject_idx_start].append(object_idx_start)
            # 数据集中主体和客体是颠倒的，数据集中的主体是唯一的，这里颠倒一下，这样每行最多只有一个关系
            # predicate_value_list[object_idx_start].append(predicate_value)
            # predicate_location_list[object_idx_start].append(subject_idx_start)

        # 把 predicate_value_list和predicate_location_list空余位置填充满
        for idx in range(len(text_tokened)):
            if len(predicate_value_list[idx]) == 0:
                predicate_value_list[idx].append("N")  # 没有关系的位置，用“N”填充
            if len(predicate_location_list[idx]) == 0:
                predicate_location_list[idx].append(idx)  # 没有关系的位置，用自身的序号填充
        
        return labeling_list, predicate_value_list, predicate_location_list, have_error

    def get_rid_unkonwn_word(self, text):
        text_rid = []
        for token in text:  # 删除不在vocab里面的词汇
            if token in self.token2id.keys():
                text_rid.append(token)
        return text_rid
    
    def get_type_rel2id(self):
        self.token_type2id = {}
        for i, token_type in enumerate(self.config.token_types):
            self.token_type2id[token_type] = i
        
        self.rel2id = {}
        for i, rel in enumerate(self.config.relations):
            self.rel2id[rel] = i
        # with open('token_type2id.json', 'w', encoding='utf-8') as f:
        #     json.dump(self.token_type2id, f, ensure_ascii=False)
        # with open('rel2id.json', 'w', encoding='utf-8') as f:
        #     json.dump(self.rel2id, f, ensure_ascii=False)
        self.token2id = {}
        with open(self.config.vocab_file, 'r', encoding='utf-8') as f:
            cnt = 0
            for line in f:
                line = line.rstrip().split()
                self.token2id[line[0]] = cnt
                cnt += 1
        self.token2id[' '] = cnt
    
    def get_data(self, file_path, is_test=False):
        data = []
        cnt = 0
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                cnt += 1
                if cnt > self.config.num_sample:
                    break
                data_item = json.loads(line)
                if not is_test:
                    spo_list = data_item['spo_list']
                else:
                    spo_list = []
                text = data_item['text']
                text_tokened = [c.lower() for c in text]  # 中文使用简单的分词
                token_type_list, predict_rel_list, predict_location_list, token_type_origin = None, None, None, None
                
                text_tokened = self.get_rid_unkonwn_word(text_tokened)
                if not is_test:
                    token_type_list, predict_rel_list, predict_location_list, have_error = self.subject_object_labeling(
                        spo_list=spo_list, text_tokened=text_tokened
                    )
                    token_type_origin = token_type_list  # 保存没有数值化前的token_type
                    if have_error:
                        continue
                item = {'text_tokened': text_tokened, 'token_type_list': token_type_list,
                        'predict_rel_list': predict_rel_list, 'predict_location_list': predict_location_list}
                # print(self.token2id[' '])
                item['text_tokened'] = [self.token2id[x] for x in item['text_tokened']]
                if not is_test:
                    item['token_type_list'] = [self.token_type2id[x] for x in item['token_type_list']]
                    # item['predict_rel_list'] = [self.rel2id[x] for x in item['predict_rel_list']]
                    predict_rel_id_tmp = []
                    for x in item['predict_rel_list']:
                        rel_tmp = []
                        for y in x:
                            rel_tmp.append(self.rel2id[y])
                        predict_rel_id_tmp.append(rel_tmp)
                    item['predict_rel_list'] = predict_rel_id_tmp
                # item['rel_predct_matrix' ] = self._get_multiple_predicate_matrix(item['predict_rel_list'],
                #                                                                  item['predict_location_list'],
                #                                                                  )
                item['text'] = ''.join(text_tokened)  # 保存消除异常词汇的文本
                item['spo_list'] = data_item['spo_list']
                item['token_type_origin'] = token_type_origin
                data.append(item)
                # data.append(data_item['text'])
                # data.append(data_item['spo_list'])
        # print(len(data))
        dataset = Dataset(data)
        if is_test:
            dataset.is_test = True
        data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.config.batch_size,
            collate_fn=dataset.collate_fn,
            drop_last=True
        )
        return data_loader
         
    def get_train_dev_data(self, path_train, path_dev=None, path_test=None):
        train_loader, dev_loader, test_loader = None, None, None
        if path_train is not None:
            train_loader = self.get_data(path_train)
        if path_dev is not None:
            dev_loader = self.get_data(path_dev)
        if path_test is not None:
            test_loader = self.get_data(path_test, is_test=True)
        
        return train_loader, dev_loader, test_loader
        

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = copy.deepcopy(data)
        self.is_test = False
    
    def __getitem__(self, index):
        text_tokened = self.data[index]['text_tokened']
        token_type_list = self.data[index]['token_type_list']
        predict_rel_list = self.data[index]['predict_rel_list']
        predict_location_list = self.data[index]['predict_location_list']
        
        data_info = {}
        for key in self.data[0].keys():
            # try:
            #     data_info[key] = locals()[key]
            # except KeyError:
            #     print('{} cannot be found in locals()'.format(key))
            if key in locals():
                data_info[key] = locals()[key]

        data_info['text'] = self.data[index]['text']
        data_info['spo_list'] = self.data[index]['spo_list']
        data_info['token_type_origin'] = self.data[index]['token_type_origin']
        return data_info
    
    def __len__(self):
        return len(self.data)
    
    def _get_multiple_predicate_matrix(self, predict_rel_list_batch, predict_location_list_batch, max_seq_length):
        batch_size = len(predict_rel_list_batch)
        predict_rel_matrix = torch.zeros((batch_size, max_seq_length, max_seq_length), dtype=torch.int64)
        for i, predict_rel_list in enumerate(predict_rel_list_batch):
            for xi, predict_rels in enumerate(predict_rel_list):
                if 0 in predict_rels:  # 0 代表是 关系 N，就是没有关系
                    continue
                for xj, predict_rel in enumerate(predict_rels):
                    object_loc = predict_location_list_batch[i][xi][xj]
                    predict_rel_matrix[i][xi][object_loc] = predict_rel
        
        return predict_rel_matrix
      
    def collate_fn(self, data_batch):
        
        def merge(sequences, is_two=False):
            lengths = [len(seq) for seq in sequences]
            max_length = max(lengths)
            # padded_seqs = torch.zeros(len(sequences), max_length)
            if is_two:
                max_len = 0
                for i, seq in enumerate(sequences):
                    for x in seq:
                        max_len = max(max_len, len(x))
                padded_seqs = torch.zeros(len(sequences), max_length, max_len)
                mask_tokens = None
            else:
                padded_seqs = torch.zeros(len(sequences), max_length)
                tmp_pad = torch.ones(1, max_length)
                mask_tokens = torch.zeros(len(sequences), max_length)
            for i, seq in enumerate(sequences):
                end = lengths[i]
                # seq = np.array(seq)
                # seq = seq.astype(float)
                if is_two:
                    # max_len = 0
                    # for x in seq:
                    #     max_len = max(max_len, len(x))
                    # padded_seqs = torch.zeros(len(sequences), max_length, max_len)
                    for j, x in enumerate(seq):
                        lenx = len(x)
                        padded_seqs[i, j, :lenx] = torch.Tensor(x)[:lenx]
                    
                else:
                    
                    # padded_seqs = torch.zeros(len(sequences), max_length)
                    seq = torch.LongTensor(seq)
                    if len(seq) != 0:
                        padded_seqs[i, :end] = seq[:end]
                        mask_tokens[i, :end] = tmp_pad[0, :end]
                    
                # seq = torch.from_numpy(seq)
                # if len(seq) != 0:
                #     padded_seqs[i, :end, :] = seq
            return padded_seqs, mask_tokens
        item_info = {}
        for key in data_batch[0].keys():
            item_info[key] = [d[key] for d in data_batch]
        token_type_list, predict_rel_list, pred_rel_matrix, predict_location_list = None, None, None, None
        text_tokened, mask_tokens = merge(item_info['text_tokened'])
        if not self.is_test:
            token_type_list, _ = merge(item_info['token_type_list'])
            predict_rel_list, _ = merge(item_info['predict_rel_list'], is_two=True)
            predict_location_list, _ = merge(item_info['predict_location_list'], is_two=True)
            max_seq_length = max([len(x) for x in text_tokened])
            pred_rel_matrix = self._get_multiple_predicate_matrix(item_info['predict_rel_list'],
                                                                  item_info['predict_location_list'],
                                                                  max_seq_length)
        # tmp = np.array(pred_rel_matrix)
        # np.savetxt('rel_matrix.txt', pred_rel_matrix[0])
        # convert to contiguous and cuda
        if USE_CUDA:
            text_tokened = text_tokened.contiguous().cuda()
            mask_tokens = mask_tokens.contiguous().cuda()
        else:
            text_tokened = text_tokened.contiguous()
            mask_tokens = mask_tokens.contiguous()

        if not self.is_test:
            if USE_CUDA:
                token_type_list = token_type_list.contiguous().cuda()
                predict_rel_list = predict_rel_list.contiguous().cuda()
                predict_location_list = predict_location_list.contiguous().cuda()
                pred_rel_matrix = pred_rel_matrix.contiguous().cuda()

            else:
                token_type_list = token_type_list.contiguous()
                predict_rel_list = predict_rel_list.contiguous()
                predict_location_list = predict_location_list.contiguous()
                pred_rel_matrix = pred_rel_matrix.contiguous()

        data_info = {'pred_rel_matrix': pred_rel_matrix, "mask_tokens": mask_tokens.to(torch.uint8)}
        data_info['text'] = item_info['text']
        data_info['spo_list'] = item_info['spo_list']
        data_info['token_type_origin'] = item_info['token_type_origin']
        for key in item_info.keys():
            # try:
            #     data_info[key] = locals()[key]
            # except KeyError:
            #     print('{} cannot be found in locals()'.format(key))
            if key in locals():
                data_info[key] = locals()[key]
        
        return data_info
        
# def _cuda(x):
#     if torch.cuda.is_available():
#         return x.cuda()
#     else:
#         return x


if __name__ == '__main__':
    config = Config()
    process = ModelDataPreparation(config)
    train_loader, dev_loader, test_loader = process.get_train_dev_data('../data/small.json')
    # train_loader, dev_loader, test_loader = process.get_train_dev_data('../data/train_data_small.json')
    print(train_loader)
    for item in train_loader:
        # print(type(item['token_type_list'][0]))
        # print(item['token_type_list'][0].shape)
        #
        # print(item['predict_rel_list'][0])
        # print(type(item['predict_rel_list'][0][0]))
        # print(item['predict_rel_list'][0].shape)
        for i in range(item['pred_rel_matrix'].shape[0]):
            for j in range(item['pred_rel_matrix'].shape[1]):
                for k in range(item['pred_rel_matrix'].shape[2]):
                    if item['pred_rel_matrix'][i, j, k] > 0.1 or item['pred_rel_matrix'][i, j, k] < -0.1:
                        print(i, j, k)
                        print(item['pred_rel_matrix'][i, j, k])
                        print("yes man")