# MedNER


## Intorduction:
It's almost a self-practice (｡･ω･｡)ﾉ . 
I've collected, combined, cleaned, and trained a Chinese medical named entity recognition dataset through the basic open-sourced bert-chinese model.

<br>

## File Dependency:
```
bert_codes -> __init__.py
            | modeling.py
            | optimization.py
            | tokenization.py
            | utils.py
            
check_points -> your_trained_model_name -> best_model.pth (your trained bert model)
                                         | log.txt (show results of each epoch)
                                         | log_dev.txt (show detailed results of each batch)
                                         | setting.txt (show hyper-parameters configuration)

datasets -> your_train_data.json
          | your_dev_data.json
          | your_test_data.json
          | your_train_features.json (generated from your_train_data.json)
          | your_dev_features.json (generated from your_dev_data.json)
          | your_test_features.json (generated from your_test_data.json)

pretrained_models -> bert_chinese -> bert_config.json
                                   | pytorch_model.pth
                                   | vocab.txt

prepro.py (create your train/dev/test datasets from the original one)

train.py (train/dev/save your model)
```

<br>

## Dataset
* **original corpora**: The original corpora used here are from [Chinese_medical_NLP](https://github.com/lrs1353281004/Chinese_medical_NLP). <br>

* **raw corpus**: Then, some of the original corpora are selected and cleaned. Please run [prepro.py](https://github.com/Schlampig/MedNER/blob/main/prepro.py) to further generate the train/dev/test data from the raw corpus. The raw corpus could be download from [here](https://pan.baidu.com/s/1_n8OczxavRmJNdXDxLgBkg) with code=xyt1. Maybe you should change the path when running the script :)<br>

* **train/dev/test data**: Sample with the following format is suitable for MedNER (you can construct your own datasets):
```
sample = {  "text": "患者3月前因“直肠癌”于在我院于全麻上行直肠癌根治术(dixon术),手术过程顺利,术后给予抗感染及营养支持治疗,患者恢复好,切口愈合良好……（略）……近期患者精神可,饮食可,大便正常,小便正常,近期体重无明显变化。",
            "entities": [
                {
                    "entity": "直肠癌",
                    "label": "诊断",
                    "sub_label": "疾病和诊断",
                    "idx_start": 8,
                    "idx_end": 11
                },
                {
                    "entity": "直肠癌根治术(dixon术)",
                    "label": "治疗",
                    "sub_label": "手术",
                    "idx_start": 21,
                    "idx_end": 35
                },
                {
                    "entity": "直肠腺癌(中低度分化),浸润溃疡型",
                    "label": "诊断",
                    "sub_label": "疾病和诊断",
                    "idx_start": 78,
                    "idx_end": 95
                },
                ..., 
                    "entity": "亚叶酸钙",
                {
                    "entity": "腹胀",
                    "label": "症状",
                    "sub_label": "症状",
                    "idx_start": 314,
                    "idx_end": 316
                },
                {
                    "entity": "直肠癌术后",
                    "label": "诊断",
                    "sub_label": "疾病和诊断",
                    "idx_start": 342,
                    "idx_end": 347
                }
            ]
        }
```

<br>

## Command Line:
* **preprocessing**: 
```bash
python prepro.py
```
* **training, evaluating, and saving the (optimal) model**:
```bash
python train.py
```
* **predicting test samples in batches**:
```bash
python predict.py
```

<br>

## Requirements
  * Python = 3.6.9
  * pytorch = 1.2.0
  * tqdm = 4.39.0
  * ipdb = 0.12.2 (optional)

<br>

## References
* **code**: the original BERT-related codes are from [bert_cn_finetune](https://github.com/ewrfcas/bert_cn_finetune) project of [ewrfcas](https://github.com/ewrfcas) and [transformers](https://github.com/huggingface/transformers) project of [Hugging Face](https://github.com/huggingface). <br>

<br>

<img src="https://github.com/Schlampig/Knowledge_Graph_Wander/blob/master/content/daily_ai_paper_view.png" height=25% width=25% />
