# EntityRelationExtraction

#### 项目说明

项目使用pytorch实现实体关系抽取中的流水线式模型。

命名实体识别部分使用的是BiLSTM+CRF。

实体关系抽取使用的是Bert进行关系分类。

最终的效果比较好。

#### 数据集说明

百度的DUIE数据集是业界规模最大的中文信息抽取数据集。它包含了43万三元组数据、21万中文句子。

句子的平均长度为54，每句话中的三元组数量的平均值为2.1。
下面是一个样本：
{"text": "据了解，《小姨多鹤》主要在大连和丹东大梨树影视城取景，是导演安建继《北风那个吹》之后拍摄的又一部极具东北文化气息的作品", 
  "spo_list": [{
  "predicate": "导演",
  "object_type": "人物",
  "subject_type": "影视作品",
  "object": "安建",
  "subject": "小姨多鹤"
  }, {
  "predicate": "导演",
  "object_type": "人物",
  "subject_type": "影视作品",
  "object": "安建",
  "subject": "北风那个吹"
  }]
}

##### 数据集和预训练模型的下载
数据集的下载
链接：https://pan.baidu.com/s/1XK3v6BQlnsvhGxgg-71IpA 
提取码：nlp0 

预训练模型和相关文件见
https://huggingface.co/bert-base-chinese/tree/main
#### 使用说明

##### 训练
Bert模型使用的是huggingface的Bert-base模型

命名实体部分的训练，直接运行mains/train_ner.py

关系抽取部分的训练，直接运行mains/train_rel.py

train.py,config.py是之前联合抽取的代码，在这个项目作废。

##### 测试

直接运行deploy/demo.py。会首先进行命名实体识别，然后将实体两两组成实体对进行关系分类。

[![Star History Chart](https://api.star-history.com/svg?repos=Xie-Minghui/EntityRelationExtraction&type=Timeline)](https://star-history.com/#Xie-Minghui/EntityRelationExtraction&Date)
