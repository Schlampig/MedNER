# MedNER

<br>

[![996.icu](https://img.shields.io/badge/link-996.icu-red.svg)](https://996.icu)
[![LICENSE](https://img.shields.io/badge/license-Anti%20996-blue.svg)](https://github.com/996icu/996.ICU/blob/master/LICENSE)

## Introduction:
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

## Examples of Results
* **format**:
```
Sent: input sentence (string).
True: true results (>entity_type: entity_mention(begin_index, end_index);...;).
Pred: predicted results (>entity_type: entity_mention(begin_index, end_index);...;).
```
* **examples**: *Note that the indices of the word are correlated to the input_ids list, not aligned to the Sent*
```
Sent: ③糖盐水：白开水纠正脱水+蔗糖10g+细盐1.75g。 
True: >部位: 盐(19, 20); >检查: 纠正脱水(9, 13); >药物: 盐水(3, 5);  
Pred: >症状: 脱水(11, 13); >药物: 糖盐水(2, 5);  

Sent: 妊娠期高血压@应详细问及患者的个人史及家族史，明确心血管疾病或其危险因素（如糖尿病、脂质异常）的相关证据。 
True: >病史: 妊娠(1, 3); 家族史(20, 23); 脂质异常(43, 47); >部位: 、(42, 43); >诊断: 高血压(4, 7); 心血管疾病(26, 31); 糖尿病(39, 42);  
Pred: >病史: 妊娠(1, 3); 家族史(20, 23); 脂质异常(43, 47); >诊断: 高血压(4, 7); 心血管疾病(26, 31); 糖尿病(39, 42);  

Sent: 转运中必须保持吸入no治疗no治疗不中断。 
True: >检查: 吸入no治疗(8, 13); 治疗(14, 16); >药物: no(13, 14);  
Pred: >检查: 吸入(8, 10); 治疗(11, 13); 治疗(14, 16); >药物: no(10, 11); no(13, 14);  

Sent: 图4-1婴儿与母乳头的含接（3）哺乳禁忌证：有活动性结核、急性传染病、艾滋病、严重心肾疾病、恶性肿瘤及精神病者以及接受放射性核素治疗或服用前述药物的母亲，不宜给婴儿哺乳。 
True: >病史: 药物(72, 74); >部位: 母乳头(8, 11); 、(29, 30); ，(77, 78); >诊断: 活动性结核(24, 29); 急性传染病(30, 35); 艾滋病(36, 39); 严重心肾疾病(40, 46); 恶性肿瘤(47, 51); 精神病(52, 55); >药物: 放射性核素治疗(60, 67); >人群: 婴儿(5, 7); 婴儿(81, 83);  
Pred: >病史: 药物(72, 74); >部位: 乳头(9, 11); >检查: 放射性核素治疗(60, 67); >诊断: 活动性结核(24, 29); 急性传染病(30, 35); 艾滋病(36, 39); 严重心肾疾病(40, 46); 恶性肿瘤(47, 51); 精神病(52, 55); >人群: 婴儿(5, 7); 婴儿(81, 83);  

Sent: b族链球菌感染@###胎膜早破(prom)使产妇更容易出现上行性子宫感染，是新生儿早发型感染的一种独立危险因素。 
True: >病史: 胎膜早破(12, 16); pro(17, 18); m(18, 19); >部位: 子宫(31, 33); >诊断: 感染(6, 8); 感染(33, 35); 早发型(40, 43); 感染(43, 45); >其他: 链球菌(3, 6); 新生儿(37, 40);  
Pred: >诊断: 感染(6, 8); 胎膜早破(12, 16); pro(17, 18); m(18, 19); 上行性子宫感染(28, 35); 新生儿早发型感染(37, 45); >其他: 链球菌(3, 6);  

Sent: 肿瘤发生在肢体的远、近端及组织类型与预后无明显相关。 
True: >部位: 肢体的远、近端(6, 13); 组织(14, 16); >诊断: 肿瘤(1, 3);  
Pred: >部位: 肢体(6, 8); 组织(14, 16); >诊断: 肿瘤(1, 3);  

Sent: 只要一吃甜食就牙疼，往里钻的疼是什么病，平时不疼 
True: >症状: 牙疼(8, 10); >部位: ，(10, 11);  
Pred: >症状: 牙疼(8, 10); >部位: ，(20, 21);  

Sent: 他们将病人分层次，根据右心室大小和是否伴有右心室依赖性冠脉循环，接受单独的、部分的双心室或全部的双心室修补，全部存活率为98%，并积累了许多经导管治疗的经验。 
True: >症状: 存活(57, 59); >症状_程度: 部分(39, 41); >部位: 右心室(12, 15); 右心室(22, 25); 冠脉(28, 30); ，(32, 33); 双心室(42, 45); >检查: 双心室修补(49, 54); 治疗(73, 75); >预后: 98(61, 62); %(62, 63); >器材: 导管(71, 73);  
Pred: >症状: 存活(57, 59); >症状_程度: 部分(39, 41); >部位: ，(9, 10); 右心室(12, 15); 右心室(22, 25); 冠脉(28, 30); 双心室(42, 45); >检查: 双心室修补(49, 54); 治疗(73, 75); >预后: 98(61, 62); %(62, 63); >器材: 导管(71, 73);  

Sent: 晚近香港大学神经外科专家创用脑室-上矢状窦分流术（吻合术），可避免其他分流术的缺点，交通性和非交通性脑积水病例均可采用。 
True: >部位: 其他(34, 36); >检查: 脑室-上矢状窦分流术(15, 25); 吻合术(26, 29); 分流术(36, 39); >诊断: 交通性和非交通性脑积水(43, 54); >科室: 神经外科(7, 11);  
Pred: >部位: 神经(7, 9); 其他(34, 36); 分流(36, 38); >检查: 脑室-上矢状(15, 21); >诊断: 交通性和非交通性脑积水(43, 54); >治疗: 吻合术(26, 29); >科室: 外科(9, 11);  

Sent: 早产@胎儿的呼吸运动消失亦提示分娩风险增加。早产@###未足月胎膜早破(pprom)未足月胎膜早破(pprom)是一个临床诊断，基于阴道流液史，消毒窥器在后穹隆看到羊水池则可进一步明确诊断。 
True: >症状: 增加(20, 22); >病史: 胎儿的呼吸运动消失(4, 13); >部位: 提示(14, 16); 胎(32, 33); 阴道(63, 65); 羊水(79, 81); >检查: 诊断(58, 60); 消毒(69, 71); 诊断(89, 91); >诊断: 早产(1, 3); 早产(23, 25); pp(37, 38); ro(38, 39); m(39, 40); pp(49, 50); ro(50, 51); m(51, 52); >治疗: 分娩(16, 18); >科室: 临床(56, 58);  
Pred: >症状: 消失(11, 13); 增加(20, 22); >病史: 运动(9, 11); >部位: 提示(14, 16); 阴道(63, 65); 羊水(79, 81); >检查: 呼吸(7, 9); 诊断(58, 60); 消毒(69, 71); 诊断(89, 91); >诊断: 早产(1, 3); 早产(23, 25); 未足月胎膜早破(29, 36); pp(37, 38); ro(38, 39); m(39, 40); pp(49, 50); ro(50, 51); m(51, 52); >治疗: 分娩(16, 18); >人群: 胎儿(4, 6); >科室: 临床(56, 58);  

Sent: 病情描述（发病时间、主要症状等）：一星期前感冒了有点咳嗽，后来严重了吃了急支糖浆没效...还有就是喉咙也疼，感觉有什么东西堵着。前天下楼梯滑几个台阶，应该没关系吧。 
True: >症状: 咳嗽(27, 29); >症状_程度: 严重(32, 34); >部位: 病(7, 8); 喉咙(50, 52); ，(75, 76); >诊断: 感冒(22, 24); >药物: 急支糖浆(37, 41);  
Pred: >症状: 咳嗽(27, 29); >症状_程度: 严重(32, 34); >部位: ，(29, 30); 喉咙(50, 52); >诊断: 感冒(22, 24); >药物: 急支糖浆(37, 41); >时间: 前天(65, 67);  

Sent: 甲状腺癌@*随访过程中最重要的是临床查体和甲状腺球蛋白测定。甲状腺癌@甲状腺球蛋白升高提示可能存在肿瘤复发，多数发生在颈中部或侧颈部。 
True: >症状: 升高(42, 44); >部位: 甲状腺球蛋白(22, 28); 甲状腺(36, 39); 蛋白(40, 42); 提示(44, 46); 颈部(65, 67); >检查: 随访(7, 9); 查体(19, 21); >诊断: 腺癌(3, 5); 腺癌(33, 35); 肿瘤复发(50, 54); >科室: 临床(17, 19);  
Pred: >部位: 甲状腺(36, 39); 蛋白(40, 42); 提示(44, 46); 部或侧颈部(62, 67); >检查: 随访(7, 9); 查体(19, 21); 甲状腺球蛋白测定(22, 30); >诊断: 腺癌(3, 5); 腺癌(33, 35); 肿瘤(50, 52); >预后: 复发(52, 54); >科室: 临床(17, 19);  

Sent: 早产儿应适当早期添加维生素a。 
True: >诊断: 早产(1, 3); >药物: 维生素a(11, 15); >其他: 早期(7, 9);  
Pred: >诊断: 早产(1, 3); >药物: 维生素a(11, 15); >其他: 早期(7, 9);  

Sent: 约2/3患者伴有肺部病变，症状轻重不等。 
True: >症状: 肺部病变(9, 13);  
Pred: >症状: 肺部病变(9, 13);  

Sent: 因此对常规检查不能确诊的患者可采用rflp，达到早期诊断的目的。 
True: >检查: 常规检查(4, 8); rf(18, 19); lp(19, 20); 诊断(25, 27); >其他: 早期(23, 25);  
Pred: >检查: 常规检查(4, 8); rf(18, 19); lp(19, 20); 诊断(25, 27); >其他: 早期(23, 25);  
```

<br>

## References
* **code**: the original BERT-related codes are from [bert_cn_finetune](https://github.com/ewrfcas/bert_cn_finetune) project of [ewrfcas](https://github.com/ewrfcas) and [transformers](https://github.com/huggingface/transformers) project of [Hugging Face](https://github.com/huggingface). <br>
* **extended learning**: [awesome_Chinese_medical_NLP](https://github.com/GanjinZero/awesome_Chinese_medical_NLP) / [Medical-Names-Corpus](https://github.com/wainshine/Medical-Names-Corpus) / [CBLUE](https://github.com/CBLUEbenchmark/CBLUE)

<br>

<img src="https://github.com/Schlampig/Knowledge_Graph_Wander/blob/master/content/daily_ai_paper_view.png" height=25% width=25% />
