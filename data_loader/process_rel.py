# -*- coding: utf-8 -*-

# @Author  : xmh
# @Time    : 2021/3/11 15:20
# @File    : process_rel.py

"""
file description:：

"""
import json
import torch
from utils.config_rel import Config, USE_CUDA
import copy
from pytorch_transformers import BertTokenizer


class DataPreparationRel:
    def __init__(self, config):
        self.config = config
    
    def get_data(self, file_path):
        data = []
        cnt = 0
        with open(file_path, 'r', encoding='utf-8')as f:
            for line in f:
                cnt += 1
                if cnt > self.config.num_sample:
                    break
                data_item = json.loads(line)
                spo_list = data_item['spo_list']
                text = data_item['text']
                for spo_item in spo_list:
                    subject = spo_item["subject"]
                    object = spo_item["object"]
                    relation = spo_item['predicate']
                    sentence_cls = '$'.join([subject, object, text.replace(subject, '#'*len(subject)).replace(object, '#')])

                
                item = {'sentence_cls': sentence_cls, 'relation': relation, 'text': text,
                        'subject': subject, 'object': object}
                data.append(item)
        
        dataset = Dataset(data)
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
            test_loader = self.get_data(path_test)
    
        return train_loader, dev_loader, test_loader
    
    
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = copy.deepcopy(data)
        self.is_test = False
        with open('../data/rel2id.json') as f:
            self.rel2id = json.load(f)
        vocab_file = '../bert-base-chinese/vocab.txt'
        self.bert_tokenizer = BertTokenizer.from_pretrained(vocab_file)
    
    def __getitem__(self, index):
        sentence_cls = self.data[index]['sentence_cls']
        relation = self.data[index]['relation']
        text = self.data[index]['text']
        subject = self.data[index]['subject']
        object = self.data[index]['object']
        
        data_info = {}
        for key in self.data[0].keys():
            if key in locals():
                data_info[key] = locals()[key]
        
        return data_info
    
    def __len__(self):
        return len(self.data)
    
    def collate_fn(self, data_batch):
        def merge(sequences):
            lengths = [len(seq) for seq in sequences]
            max_length = max(lengths)
            # padded_seqs = torch.zeros(len(sequences), max_length)
            padded_seqs = torch.zeros(len(sequences), max_length)
            tmp_pad = torch.ones(1, max_length)
            mask_tokens = torch.zeros(len(sequences), max_length)
            for i, seq in enumerate(sequences):
                end = lengths[i]
                seq = torch.LongTensor(seq)
                if len(seq) != 0:
                    padded_seqs[i, :end] = seq[:end]
                    mask_tokens[i, :end] = tmp_pad[0, :end]
            
            return padded_seqs, mask_tokens
        
        item_info = {}
        for key in data_batch[0].keys():
            item_info[key] = [d[key] for d in data_batch]

        # 转化为数值
        sentence_cls = [self.bert_tokenizer.encode(sentence) for sentence in item_info['sentence_cls']]
        relation = torch.Tensor([self.rel2id[rel] for rel in item_info['relation']]).to(torch.int64)
        
        # 批量数据对齐
        sentence_cls, mask_tokens = merge(sentence_cls)
        
        if USE_CUDA:
            sentence_cls = sentence_cls.contiguous().cuda()
            relation = relation.contiguous().cuda()
        else:
            sentence_cls = sentence_cls.contiguous()
            relation = relation.contiguous()

        data_info = {"mask_tokens": mask_tokens.to(torch.uint8)}
        data_info['text'] = item_info['text']
        data_info['subject'] = item_info['subject']
        data_info['object'] = item_info['object']
        for key in item_info.keys():
            if key in locals():
                data_info[key] = locals()[key]
        
        return data_info
        
        
if __name__ == '__main__':
    config = Config()
    process = DataPreparationRel(config)
    train_loader, dev_loader, test_loader = process.get_train_dev_data('../data/train_small.json')
    
    for item in train_loader:
        print(item)
    

