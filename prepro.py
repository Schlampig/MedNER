import os
import re
import json
import math
import random
from tqdm import tqdm
from pprint import pprint
import ipdb

"""
sample = {"text": text,
          "entities": [{“entity”: str, 
                          “label”: str, 
                          “sub_label”: str,
                          "idx_start": int,
                          "idx_end": int},
                          ...]
where, text[idx_start:idx_end] == entity.
"""


def unify_key():
    d_new = dict()
    d = {"症状": ["症状", "symptom", "sym", "症状"],
         "病史": ["社会学"],
         "症状_程度": ["feature"],
         "症状_生理": ["physiology"],
         "部位": ["解剖部位", "body", "bod", "部位"],
         "检查": ["影像检查", "实验室检验", "ite", "test", "pro", "检查"],
         "诊断": ["疾病和诊断", "disease", "dis", "疾病"],
         "治疗": ["手术", "treatment", "手术治疗", "其他治疗"],
         "药物": ["药物", "drug", "dru", "药物"],
         "预后": ["预后"],
         "器材": ["equ"],
         "人群": ["crowd"],
         "科室": ["department", "dep"],
         "时间": ["time"],
         "其他": ["mic", "流行病学", "其他"]}
    for k, lst_v in d.items():
        for v in lst_v:
            d_new.update({v: k})
    return d, d_new
D_RAW, D_CLASS = unify_key()


def find_index(s, w):
    try:
        idx_s, idx_e = re.search(w, s).span()
    except:
        idx_s, idx_e = None, None
    return idx_s, idx_e


def update_dict(d, k, v):
    if k in d.keys():
        d[k].append(v)
        d[k] = list(set(d[k]))
    else:
        d[k] = [v]
    return d


def check_cEHRNER(load_dir="../raw_data/ner/cEHRNER/"):
    global D_CLASS
    d_key = dict()
    lst_sample = list()
    for load_file in ["train.json", "dev.json", "test.json"]:
        print("Now operate the file - {}".format(load_file))
        load_path = os.path.join(load_dir, load_file)
        with open(load_path, "r") as f:
            for line in tqdm(f.readlines()):
                line = json.loads(line)
                text = line.get("text", None)
                lst_mention = line.get("mention_data")
                lst_entity = list()
                for d in lst_mention:
                    m_v = d.get("mention", None)
                    m_k = d.get("label", None)
                    if text and m_k and m_v:
                        d_key = update_dict(d_key, m_k, m_v)
                        sample_now = {"entity": m_v,
                                      "label": D_CLASS.get(m_k, None),
                                      "sub_label": m_k,
                                      "idx_start": int(d.get("offset")),
                                      "idx_end": int(d.get("offset")) + len(m_v)}
                        lst_entity.append(sample_now)
                lst_sample.append({"text": text, "entities": lst_entity})
    return d_key, lst_sample


def check_cMedQANER(load_dir="../raw_data/ner/cMedQANER/"):
    global D_CLASS
    d_key = dict()
    lst_sample = list()
    for load_file in ["train.json", "dev.json", "test.json"]:
        print("Now operate the file - {}".format(load_file))
        load_path = os.path.join(load_dir, load_file)
        with open(load_path, "r") as f:
            for line in tqdm(f.readlines()):
                line = json.loads(line)
                text = line.get("text", None)
                lst_mention = line.get("mention_data")
                lst_entity = list()
                for d in lst_mention:
                    m_v = d.get("mention", None)
                    m_k = d.get("type", None)
                    if text and m_k and m_v:
                        d_key = update_dict(d_key, m_k, m_v)
                        sample_now = {"entity": m_v,
                                      "label": D_CLASS.get(m_k, None),
                                      "sub_label": m_k,
                                      "idx_start": int(d.get("offset")),
                                      "idx_end": int(d.get("offset")) + len(m_v)}
                        lst_entity.append(sample_now)
                lst_sample.append({"text": text, "entities": lst_entity})
    return d_key, lst_sample


def check_chip1(load_dir="../raw_data/ner/chip1/"):
    global D_CLASS
    d_key = dict()
    lst_sample = list()
    for load_file in ["train_data.txt", "val_data.txt"]:
        print("Now operate the file - {}".format(load_file))
        load_path = os.path.join(load_dir, load_file)
        with open(load_path, "r") as f:
            for line in tqdm(f.readlines()):
                lst_line = line.split("|||")
                text = lst_line[0]
                lst_entity = list()
                for mention in lst_line[1:-1]:
                    lst_mention = mention.split()
                    idx_s, idx_e, m_k = int(lst_mention[0]), int(lst_mention[1]), lst_mention[2]
                    m_v = text[idx_s:(idx_e+1)]
                    if text and m_k and m_v:
                        d_key = update_dict(d_key, m_k, m_v)
                        sample_now = {"entity": m_v,
                                      "label": D_CLASS.get(m_k, None),
                                      "sub_label": m_k,
                                      "idx_start": idx_s,
                                      "idx_end": idx_e+1}  # note: index bias exists in this case
                        lst_entity.append(sample_now)
                lst_sample.append({"text": text, "entities": lst_entity})
    return d_key, lst_sample


def check_chip2(load_dir="../raw_data/ner/chip2/"):
    global D_CLASS
    d_key = dict()
    lst_sample = list()
    for load_file in ["train_data.json", "val_data.json"]:
        print("Now operate the file - {}".format(load_file))
        load_path = os.path.join(load_dir, load_file)
        with open(load_path, "r") as f:
            for line in tqdm(f.readlines()):
                line = json.loads(line)
                text = line.get("text", None)
                lst_mention = line.get("spo_list")
                lst_entity = list()
                for d in lst_mention:
                    m_k_1 = d.get("subject_type", None)
                    m_v_1 = d.get("subject", None)
                    m_k_2 = d.get("object_type", None).get("@value", None)
                    m_v_2 = d.get("object", {}).get("@value", None)
                    if text:
                        if m_k_1 and m_v_1:
                            d_key = update_dict(d_key, m_k_1, m_v_1)
                            idx_s, idx_e = find_index(text, m_v_1)
                            if idx_s and idx_e:
                                sample_now = {"entity": m_v_1,
                                              "label": D_CLASS.get(m_k_1, None),
                                              "sub_label": m_k_1,
                                              "idx_start": idx_s,
                                              "idx_end": idx_e}
                                lst_entity.append(sample_now)
                        if m_k_2 and m_v_2:
                            d_key = update_dict(d_key, m_k_2, m_v_2)
                            idx_s, idx_e = find_index(text, m_v_2)
                            if idx_s and idx_e:
                                sample_now = {"entity": m_v_2,
                                              "label": D_CLASS.get(m_k_2, None),
                                              "sub_label": m_k_2,
                                              "idx_start": idx_s,
                                              "idx_end": idx_e}
                                lst_entity.append(sample_now)
                lst_sample.append({"text": text, "entities": lst_entity})
    return d_key, lst_sample


def check_YiduS4K(load_dir="../raw_data/ner/YiduS4K/"):
    global D_CLASS
    d_key = dict()
    lst_sample = list()
    for load_file in ["subtask1_training_part1.txt", "subtask1_training_part2.txt", "subtask1_test_set_with_answer.txt"]:
        print("Now operate the file - {}".format(load_file))
        load_path = os.path.join(load_dir, load_file)
        with open(load_path, "r") as f:
            for line in tqdm(f.readlines()):
                line = line.encode('utf8').decode('utf-8-sig')
                try:
                    line = json.loads(line)
                except:
                    continue
                text = line.get("originalText", None)
                lst_mention = line.get("entities")
                lst_entity = list()
                for d in lst_mention:
                    idx_s, idx_e = d.get("start_pos", None), d.get("end_pos", None)
                    m_v = text[idx_s: idx_e] if text and idx_s and idx_e else None
                    m_k = d.get("label_type", None)
                    if text and m_k and m_v:
                        d_key = update_dict(d_key, m_k, m_v)
                        sample_now = {"entity": m_v,
                                      "label": D_CLASS.get(m_k, None),
                                      "sub_label": m_k,
                                      "idx_start": idx_s,
                                      "idx_end": idx_e}  # note: index bias exists in this case
                        lst_entity.append(sample_now)
                lst_sample.append({"text": text, "entities": lst_entity})
    return d_key, lst_sample


def check_label():
    print("******************** Check original datasets' labels ******************** ")

    def pretty_str(s):
        s = str(s)
        return s.split("at")[0].replace("<function check_", "").strip()

    lst_f = [check_cEHRNER, check_cMedQANER, check_chip1, check_chip2, check_YiduS4K]
    lst = list()
    for f in lst_f:
        d = f()[0]
        for d_k, d_v in d.items():
            lst.append([pretty_str(f), d_k, d_v[0], d_v[1], d_v[2], d_v[3], d_v[4]])
    for i in lst:
        print("{}, \t key: {}, \t value: {}, {}, {}, {}, {}".format(i[0], i[1], i[2], i[3], i[4], i[5], i[6]))
    return None


def combine_dict(save_path="ner_all.json"):
    print("******************** Combine the original datasets ******************** ")
    lst_f = [check_cEHRNER, check_cMedQANER, check_chip1, check_chip2, check_YiduS4K]
    lst = list()
    for f in tqdm(lst_f):
        lst.extend(f()[1])
    if isinstance(save_path, str) and save_path.endswith(".json"):
        with open(save_path, 'w') as f:
            json.dump(lst, f, ensure_ascii=False, indent=4)
        print("Succeed to save, 5 examples are shown below:")
        pprint(lst[-5:])
    else:
        print("Fail to save, wrong saving path.")
    return lst


def statistic_info(data_now=None):
    print("******************** Statistic Info about the combined dataset ******************** ")
    global D_RAW
    # load file
    if isinstance(data_now, str) and data_now.endswith(".json"):
        with open(data_now, "r") as f:
            lst = json.load(f)
    elif isinstance(data_now, list):
        lst = data_now
    else:
        return None, None, None, None
    # initialize dictionary
    d_type = {k: 0 for k, v in D_RAW.items()}
    d_type.update({"未知": 0})
    avg_text_len = 0
    # count
    size_sample = len(lst)
    size_entity = sum(len(d.get("entities")) for d in lst)
    for i_sample, sample in enumerate(lst):
        avg_text_len += len(sample.get("text", ""))
        for d in sample.get("entities"):
            try:
                d_type[d.get("label")] += 1
            except:
                d_type["未知"] += 1
    avg_text_len = avg_text_len/float(i_sample)
    print("Distribution of labels:")
    pprint(d_type)
    print()
    print("Sample Size: {}| Entity Size: {}| Average Text Length: {:.2f}".format(size_sample, size_entity, avg_text_len))
    return size_sample, size_entity, d_type, avg_text_len


def prettify_data(data_now=None, save_path="ner_all_new.json"):
    print("******************** Prettify the combined dataset ******************** ")
    # load file
    if isinstance(data_now, str) and data_now.endswith(".json"):
        with open(data_now, "r") as f:
            lst = json.load(f)
    elif isinstance(data_now, list):
        lst = data_now
    else:
        return None, None

    # get dict_entity = {mention:{mention_label_1: count, mention_label_2: count, ...}}
    print("Generate dict_entity ...")
    dict_entity = dict()
    for sample in tqdm(lst):
        entities = sample.get("entities")
        for d in entities:
            entity_now, label_now = d.get("entity"), d.get("label")
            if entity_now in dict_entity.keys():
                if label_now in dict_entity[entity_now].keys():
                    dict_entity[entity_now][label_now] += 1
                else:
                    dict_entity[entity_now][label_now] = 1
            else:
                dict_entity[entity_now] = {label_now: 1}

    # dict_entity -> {mention: mention_label, ...}
    for d_k, d_v in dict_entity.items():
        dict_entity[d_k] = max(d_v, key=lambda k: d_v[k])

    # prettify entity mention in data (add missing mention / correct label)
    print("Update the original data ...")
    for i_sample, sample in enumerate(tqdm(lst)):
        text = sample.get("text")
        entities = sample.get("entities")
        text_tag = "0" * len(text)
        for i_d, d in enumerate(entities):
            if dict_entity.get(d.get("entity"), None):
                lst[i_sample]["entities"][i_d]["label"] = dict_entity[d.get("entity")]  # unify label
            text_tag = text_tag[:d.get("idx_start")] + \
                       "1" * (d.get("idx_end") - d.get("idx_start")) + \
                       text_tag[d.get("idx_end"):]  # record  tag

        for mention, mention_label in dict_entity.items():
            try:
                iter_match = re.finditer(mention, text)
            except:
                continue
            for match in iter_match:
                if match.span():
                    if "1" in text_tag[match.start(): match.end()]:  # find and add the new entity
                        continue
                    else:
                        text_tag = text_tag[:match.start()] + "1" * (match.end() - match.start()) + text_tag[match.end():]
                        entity_new = {"entity": mention,
                                      "label": mention_label,
                                      "sub_label": "新增术语",
                                      "idx_start": match.start(),
                                      "idx_end": match.end()}
                        lst[i_sample]["entities"].append(entity_new)

    # save updated lst
    print("Save the updated data ...")
    if isinstance(save_path, str) and save_path.endswith(".json"):
        with open(save_path, 'w') as f:
            json.dump(lst, f, ensure_ascii=False, indent=4)
        print("Succed to save.")
    else:
        print("Fail to save, wrong saving path.")
    return lst, dict_entity


def remove_one_token(data_now=None):
    print("******************** Remove token with length=1 (Optional) ******************** ")
    # load file
    if isinstance(data_now, str) and data_now.endswith(".json"):
        with open(data_now, "r") as f:
            lst = json.load(f)
    elif isinstance(data_now, list):
        lst = data_now
    else:
        return None
    # delete one-char entities
    count_one = 0
    count_all = 0
    for i_sample, sample in enumerate(tqdm(lst)):
        for i_d, d in enumerate(sample["entities"]):
            count_all += 1
            if len(d["entity"]) == 1:
                count_one += 1
                lst[i_sample]["entities"].pop(i_d)
    # save
    save_path = data_now if isinstance(data_now, str) and data_now.endswith(".json") else "ner_delete.json"
    with open(save_path, 'w') as f:
        json.dump(lst, f, ensure_ascii=False, indent=4)
    print("Count_one / Count_all = {} / {} \n".format(count_one, count_all))
    print("Succeed to save.")
    return lst


def get_tiny_data(data_now=None, size=100, save_path="ner_tiny.json"):
    print("******************** Create Tiny_Data by selecting from the prettified datasets (Optional)  ******************** ")
    # load file
    if isinstance(data_now, str) and data_now.endswith(".json"):
        with open(data_now, "r") as f:
            lst = json.load(f)
    elif isinstance(data_now, list):
        lst = data_now
    else:
        return False
    # cut the data
    lst = lst[:min(size, len(lst))]
    # save cut lst
    print("Save the updated data ...")
    if isinstance(save_path, str) and save_path.endswith(".json"):
        with open(save_path, 'w') as f:
            json.dump(lst, f, ensure_ascii=False, indent=4)
        print("Succeed to save.")
    else:
        print("Fail to save, wrong saving path.")
    return True


def split_train_val_test(data_now=None, ratio=0.8, seed=42):
    print("******************** Split datasets into Train/Dev/Test ******************** ")
    global D_RAW
    assert 0 < ratio < 1
    # load file
    if isinstance(data_now, str) and data_now.endswith(".json"):
        with open(data_now, "r") as f:
            lst = json.load(f)
    elif isinstance(data_now, list):
        lst = data_now
    else:
        return False
    # shuffle the list
    random.seed(seed)
    random.shuffle(lst)
    # split the list
    idx = math.ceil(len(lst) * ratio)
    train_data, rest_data = lst[:idx], lst[idx:]
    dev_data = rest_data[:math.ceil(0.5*len(rest_data))]
    test_data = rest_data[math.ceil(0.5 * len(rest_data)):]
    print("Sample numbers: train-{}, dev-{}, test-{}".format(len(train_data), len(dev_data), len(test_data)))
    # save the file
    with open("ner_train.json", 'w') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=4)
    with open("ner_dev.json", 'w') as f:
        json.dump(dev_data, f, ensure_ascii=False, indent=4)
    with open("ner_test.json", 'w') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=4)
    print("Succeed to save.")
    return True


if __name__ == "__main__":
    # check_label()
    # lst = combine_dict()
    # size_sample, size_entity, d_type, avg_text_len = statistic_info(data_now="datasets/ner_all.json")

    # lst, dict_entity = prettify_data(data_now="datasets/ner_all.json")
    # remove_one_token(data_now="datasets/ner_all.json")
    
    # get_tiny_data(data_now="datasets/ner_all.json")
    # split_train_val_test(data_now="datasets/ner_all.json")
    
    print("Finished.")
