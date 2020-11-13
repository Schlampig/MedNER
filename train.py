import os
import copy
import json
import random
import argparse
import numpy as np
from time import time
from tqdm import tqdm
import torch
from torch.utils.data import TensorDataset, DataLoader
from bert_codes.modeling import BertForTokenClassification, BertConfig
from bert_codes.optimization import AdamW, get_linear_schedule_with_warmup
import bert_codes.tokenization as tokenization
import bert_codes.utils as utils
import ipdb

# Configuration
##############################################################################################
DICT_LABEL = {"症状": [1, 2], "病史": [3, 4], "症状_程度": [5, 6], "症状_生理": [7, 8], "部位": [9, 10],
              "检查": [11, 12], "诊断": [13, 14], "治疗": [15, 16], "药物": [17, 18], "预后": [19, 20],
              "器材": [21, 22], "人群": [23, 24], "科室": [25, 26], "时间": [27, 28], "其他": [29, 30]}
DICT_LABEL_REV = dict()
for k, v in DICT_LABEL.items():
    DICT_LABEL_REV.update({v[0]: k})
    DICT_LABEL_REV.update({v[1]: k})

t_config = time()
parser = argparse.ArgumentParser()
parser.add_argument('--gpu_ids', type=str, default='2, 3')
# training parameter
parser.add_argument('--train_epochs', type=int, default=25)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--dev_batch_size', type=int, default=128)
parser.add_argument('--max_seq_length', type=int, default=128)
parser.add_argument('--max_lines', type=str, default=-1)  # number of lines readed from the raw text
parser.add_argument('--lr', type=float, default=5e-5)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--clip_norm', type=float, default=1.0)
parser.add_argument('--warmup_rate', type=float, default=0.1)
parser.add_argument("--schedule", default='warmup_linear', type=str, help='schedule')
parser.add_argument("--weight_decay_rate", default=0.01, type=float, help='weight_decay_rate')
parser.add_argument('--float16', type=bool, default=True)  # only sm >= 7.0 (tensorcores)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--log_interval', type=int, default=50)
parser.add_argument('--vocab_size', type=int, default=21128)  # according to the vocab file information
# data and model dir
parser.add_argument('--train_dir', type=str, default='./datasets/ner_train.json')
parser.add_argument('--dev_dir', type=str, default='./datasets/ner_dev.json')
parser.add_argument('--feature_train_dir', type=str, default='./datasets/fea_ner_train.json')
parser.add_argument('--feature_dev_dir', type=str, default='./datasets/fea_ner_dev.json')
parser.add_argument('--bert_config_file', type=str, default='./pretrained_models/bert_chinese/bert_config.json')
parser.add_argument('--init_restore_dir', type=str, default='./pretrained_models/bert_chinese/pytorch_model.pth')
parser.add_argument('--vocab_file', type=str, default='./pretrained_models/bert_chinese/vocab.txt')
parser.add_argument('--checkpoint_dir', type=str, default='check_points/base_ner')
parser.add_argument('--setting_file', type=str, default='setting.txt')
parser.add_argument('--log_file', type=str, default='log.txt')
parser.add_argument('--log_dev_file', type=str, default='log_dev.txt')
# set args
args = parser.parse_args()
utils.check_args(args)

# bert initialization
bert_config = BertConfig.from_json_file(args.bert_config_file)
bert_config.attention_probs_dropout_prob = args.dropout
bert_config.hidden_dropout_prob = args.dropout
tokenizer = tokenization.BertTokenizer(vocab_file=args.vocab_file, do_lower_case=True)
model = BertForTokenClassification(bert_config, num_labels=len(DICT_LABEL_REV)+1)

# set seed
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# set gpu
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
device = torch.device("cuda")
n_gpu = torch.cuda.device_count()
if n_gpu > 0:
    torch.cuda.manual_seed_all(args.seed)
print("device %s n_gpu %d" % (device, n_gpu))
print("device: {} n_gpu: {} 16-bits training: {}".format(device, n_gpu, args.float16))

# set model
print('***** Initializing Model *****')
utils.torch_init_model(model, args.init_restore_dir)
model.to(device)

print("Configuration Time: {}".format(time() - t_config))


# Data Pre-processing
##############################################################################################
def line2sample(line):
    global args, DICT_LABEL
    # text and entities obtaining
    text = line.get("text", "")  # text is a string
    lst_entities = line.get("entities", [])
    if len(text) == 0 or len(lst_entities) == 0:
        return None
    # tagging
    lst_tag = [0] * len(text)
    for entity in lst_entities:
        idx_s, idx_e, label = entity.get("idx_start"), entity.get("idx_end"), entity.get("label")
        lst_tag[idx_s] = DICT_LABEL.get(label, [29, 30])[0]
        lst_tag[(idx_s+1):idx_e] = [DICT_LABEL.get(label, [29, 30])[1]]*(idx_e - idx_s - 1)  # lst_tag = [0, 0, 1, 2, 2, 0, ...]
    # pre-splitting and aligning: "abcdefghijk" with [0, 0, 1, 2, 0, 0, 0, 3, 4, 4, 0] -> [a, b, cd, e, fg, hij, k] with [0, 0, 1, 0, 0, 3, 0]
    lst_tag_new = list()
    lst_text = utils.split_sent(text)  # lst_text = [word, word, word, ...]
    idx_now = 0
    for text_now in lst_text:
        idx_new = text[idx_now:].find(text_now)  # find start index
        idx_now += idx_new
        lst_tag_new.extend([lst_tag[idx_now]])  # lst_tag_new = [0, 0, 1, 2, 2, 0, ...]
    # tokenizing
    lst_token = list()  # lst_token = [token, token, token, ...]
    input_tags = list()
    for i_text, text_now in enumerate(lst_text):
        text_new = tokenizer.tokenize(text_now)
        lst_token.extend(text_new)
        input_tags.extend([lst_tag_new[i_text]]*len(text_new))
    # token to ids
    if len(lst_token) > args.max_seq_length - 2:  # 2 means [CLS] and [SEP]
        lst_token = lst_token[:args.max_seq_length - 2]
        input_tags = input_tags[:args.max_seq_length - 2]
    lst_token = ["[CLS]"] + lst_token + ["[SEP]"]
    input_tags = [0] + input_tags + [0]
    input_ids = tokenizer.convert_tokens_to_ids(lst_token)  # token to ids
    # input_mask and input_segments
    input_mask = [1] * len(input_ids)
    input_segments = [0] * len(input_ids)
    # padding
    while len(input_ids) < args.max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        input_segments.append(0)
        input_tags.append(0)
    # generate sample
    sample = {"input_ids": input_ids,
           "input_mask": input_mask,
           "input_segments": input_segments,
           "input_tags": input_tags}
    return sample


def data2fea(load_path, save_path):
    global args
    # load/create file
    if os.path.exists(save_path):
        print("***** Loading Feature Data *****")
        with open(save_path, "r") as f:
            features = json.load(f)
    else:
        print("***** Creating Feature Data *****")
        features = list()
        with open(load_path, "r") as f:
            lst_data = json.load(f)
            i_sample = 0
            for line in tqdm(lst_data):
                if args.max_lines > 0:
                    if i_sample > args.max_lines:
                        break
                sample = line2sample(line)
                if sample:
                    features.append(sample)
                    sample.update({"id": i_sample})
                    i_sample += 1
    # save file
    print("***** Saving Feature Data *****")
    with open(save_path, "w") as f:
        json.dump(features, f)
    return features


def prepare_for_train():
    global args
    # get features
    train_features = data2fea(load_path=args.train_dir, save_path=args.feature_train_dir)
    dev_features = data2fea(load_path=args.dev_dir, save_path=args.feature_dev_dir)
    # get train dataloader
    train_input_ids = torch.tensor([f['input_ids'] for f in train_features], dtype=torch.long)
    train_input_mask = torch.tensor([f['input_mask'] for f in train_features], dtype=torch.long)
    train_input_segments = torch.tensor([f['input_segments'] for f in train_features], dtype=torch.long)
    train_input_tags = torch.tensor([f['input_tags'] for f in train_features], dtype=torch.long)
    train_tensor = TensorDataset(train_input_ids, train_input_mask, train_input_segments, train_input_tags)
    train_dataloader = DataLoader(train_tensor, batch_size=args.batch_size, shuffle=True)
    # get dev dataloader
    dev_input_ids = torch.tensor([f['input_ids'] for f in dev_features], dtype=torch.long)
    dev_input_mask = torch.tensor([f['input_mask'] for f in dev_features], dtype=torch.long)
    dev_input_segments = torch.tensor([f['input_segments'] for f in dev_features], dtype=torch.long)
    dev_input_tags = torch.tensor([f['input_tags'] for f in dev_features], dtype=torch.long)
    dev_tensor = TensorDataset(dev_input_ids, dev_input_mask, dev_input_segments, dev_input_tags)
    dev_dataloader = DataLoader(dev_tensor, batch_size=args.dev_batch_size, shuffle=False)
    print("Train-{}, Dev-{}".format(len(train_features), len(dev_features)))
    return train_features, train_dataloader, dev_features, dev_dataloader


# Evaluate
##############################################################################################
def get_entity_dict(lst):
    """
    :param: lst = [0, 0, 1, 2, 2, 0, 0, 3, 4, 4, 0, 0, 5, 6, 0, 8, 0, 9, 0, 7, 0, 0, 2, 1, 1, 2, 0, 0]
    :return: d = {'病史': ['7_9'], '症状': ['2_4', '23_23', '24_26'], '症状_生理': ['19_19'], '科室': [], '药物': [], ...}
    """
    global DICT_LABEL, DICT_LABEL_REV
    d = {k: [] for k in DICT_LABEL.keys()}
    stack = list()
    idx = 0
    for i, tag in enumerate(lst):
        if tag > 0:  # i = B or I
            if len(stack) == 1:  # [B]
                if tag == stack[-1]+1:  # [BI]
                    stack.append(tag)
                    idx = i
                else:  # pop current entity
                    s_now = str(idx - len(stack) + 1) + "_" + str(idx + 1)
                    d[DICT_LABEL_REV.get(stack[0])].append(s_now)
                    stack = [tag] if tag % 2 == 1 else []
                    idx = i
            elif len(stack) > 1:  # [BI...I]
                if tag == stack[-1]:  # [BI...II]
                    stack.append(tag)
                    idx = i
                else:  # pop current entity
                    s_now = str(idx - len(stack) + 1) + "_" + str(idx + 1)
                    d[DICT_LABEL_REV.get(stack[0])].append(s_now)
                    stack = [tag] if tag % 2 == 1 else []
                    idx = i
            else:  # []
                stack = [tag] if tag % 2 == 1 else []
                idx = i
    if len(stack) > 0:  # pop the last entity
        s_now = str(idx - len(stack) + 1) + "_" + str(idx + 1)
        d[DICT_LABEL_REV.get(stack[0])].append(s_now)
    return d


def compare_entity_dict(d_true, d_pred):
    """
    :param d_pred = {'病史': ['7_9'], '症状': ['2_4', '23_23', '24_26'], '症状_生理': ['19_19'], '科室': [], ...}
    :param d_true = {'病史': ['8_9'], '症状': ['2_3', '23_23', '24_26'], '症状_生理': ['19_19'], '科室': [], ...}
    :return: d_res = {'病史': { 'f1': 0.0, 'precision': 0.0, 'recall': 0.0},
                '症状': { 'f1': 0.67, 'precision': 0.67,'recall': 0.67},
                '症状_生理': {'f1': 0.99, 'precision': 0.99, 'recall': 0.99},
                '科室': { 'f1': 0.0, 'precision': 0.0, 'recall': 0.0}, ...}
    """
    global DICT_LABEL
    d_res = dict()
    for k in DICT_LABEL.keys():
        set_true = set(d_true.get(k))
        set_pred = set(d_pred.get(k))
        correct = len(set.intersection(set_true, set_pred))
        precision = correct / (len(set_pred) + 1e-5)
        recall = correct / (len(set_true) + 1e-5)
        f1 = (2 * precision * recall) / (precision + recall + 1e-5)
        d_res.update({k: {"precision": precision, "recall": recall, "f1": f1}})
    return d_res


def print_and_save_batch_dict(input_id, d_true, d_pred):
    global args, tokenizer
    lst_token = tokenizer.convert_ids_to_tokens(input_id.cpu().numpy())
    lst_token = [t.replace("##", "") if t.startswith("##") else t for t in lst_token]
    s_token = "".join(lst_token).replace("[PAD]", "").replace("[CLS]", "").replace("[SEP]", "")
    with open(args.log_dev_file, 'a') as aw_dev:
        aw_dev.write("Sent: {} \n".format(s_token))
    s_true, s_pred = "", ""
    for k, lst_v in d_true.items():
        if len(lst_v) > 0:
            s_true = s_true + ">" + k + ": "
            for i_m, m in enumerate(lst_v):
                lst_m = m.split("_")
                idx_s, idx_e = int(lst_m[0]), int(lst_m[1])
                s_true = s_true + "".join(lst_token[idx_s:idx_e]) + "(" + lst_m[0] + ", " + lst_m[1] + "); "
    for k, lst_v in d_pred.items():
        if len(lst_v) > 0:
            s_pred = s_pred + ">" + k + ": "
            for m in lst_v:
                lst_m = m.split("_")
                idx_s, idx_e = int(lst_m[0]), int(lst_m[1])
                s_pred = s_pred + "".join(lst_token[idx_s:idx_e]) + "(" + lst_m[0] + ", " + lst_m[1] + "); "
    with open(args.log_dev_file, 'a') as aw_dev:
        aw_dev.write("True: {} \n".format(s_true))
        aw_dev.write("Pred: {} \n".format(s_pred))
        # aw_dev.write("True: {} \n".format(d_true))
        # aw_dev.write("Pred: {} \n".format(d_pred))
        aw_dev.write("\n")
    return None
    

def evaluate(dev_dataloader):
    global model, DICT_LABEL
    print("***** Eval *****")
    model.eval()
    d_res = {k:{"precision": [], "recall": [], "f1": []} for k in DICT_LABEL.keys()}
    with torch.no_grad():
        for step, batch in enumerate(dev_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, input_segments, input_tags = batch
            logits = model(input_ids=input_ids,
                           token_type_ids=input_segments,
                           attention_mask=input_mask)
            logits = logits.detach().cpu().numpy()  # [bs, len, dim]
            pred_batch = np.argmax(logits, axis=-1)  # get predicted labels: [bs,len]
            true_batch = input_tags.detach().cpu().numpy()  # get true labels
            # calculate each sample in the batch
            batch_size = true_batch.shape[0]
            for i in range(batch_size):
                true_batch_now = get_entity_dict(true_batch[i])
                pred_batch_now = get_entity_dict(pred_batch[i])
                input_id_now = input_ids[i]
                res_batch = compare_entity_dict(d_true=true_batch_now, d_pred=pred_batch_now)
                print_and_save_batch_dict(input_id=input_id_now, d_true=true_batch_now, d_pred=pred_batch_now)
            for k, v in res_batch.items():
                d_res[k]["precision"].append(v["precision"])
                d_res[k]["recall"].append(v["recall"])
                d_res[k]["f1"].append(v["f1"])
            with open(args.log_dev_file, 'a') as aw_dev:
                aw_dev.write("Result of batch {} is: \n {} \n ".format(step, res_batch))
                aw_dev.write(" ---------------------------------------------------- \n")
            print("Result of batch {} is: \n {} \n ".format(step, res_batch))
        # get final scores
        f1 = np.mean([np.mean(v["f1"]) for k, v in d_res.items()])
    return f1


# Train
##############################################################################################
def learn(train_features=None, train_dataloader=None, dev_features=None, dev_dataloader=None):
    global args, tokenizer, model, n_gpu, device
    assert train_features and train_dataloader and dev_dataloader

    print("***** Pre-Settings *****")
    # set steps
    steps_per_epoch = len(train_features) // args.batch_size
    if len(train_features) % args.batch_size != 0:
        steps_per_epoch += 1
    total_steps = steps_per_epoch * args.train_epochs
    print("steps per epoch:", steps_per_epoch)
    print("total steps:", total_steps)
    print("warm-up steps:", int(args.warmup_rate * total_steps))

    # set optimization
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay_rate},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_rate * total_steps),
                                                num_training_steps=total_steps)
    # init amp
    if args.float16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    # do paralleling
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    print("***** Training *****")
    model.train()
    global_steps = 0
    best_score = 0.
    for i in range(int(args.train_epochs)):
        print("Starting epoch %d" % (i + 1))
        with open(args.log_dev_file, 'a') as aw_dev:
            aw_dev.write(" ---------------------------------------------------- \n")
            aw_dev.write("Epoch:{} \n".format(i + 1))
        start_time = time()
        loss_values = []
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, input_segments, input_tags = batch
            loss = model(input_ids=input_ids,
                         token_type_ids=input_segments,
                         attention_mask=input_mask,
                         labels=input_tags)
            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            loss_values.append(loss.item())
            if args.float16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            if args.float16:
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.clip_norm)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)

            # update learning rate schedule
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            global_steps += 1

            if global_steps % args.log_interval == 0:
                show_str = 'Epoch:{}, Steps:{}/{}, Loss:{:.4f}'.format(i + 1, global_steps, total_steps,
                                                                       np.mean(loss_values))
                with open(args.log_file, 'a') as aw:
                    aw.write("Epoch:{}, Steps:{}/{}, Loss:{:.4f}".format(i + 1, global_steps, total_steps,
                                                                         np.mean(loss_values)) + '\n')
                if global_steps > 1:
                    remain_seconds = (time() - start_time) * ((steps_per_epoch - step) / (step + 1e-5))
                    m, s = divmod(remain_seconds, 60)
                    h, m = divmod(m, 60)
                    remain_time = " remain:%02d:%02d:%02d" % (h, m, s)
                    show_str += remain_time
                print(show_str)

        # evaluate
        print("***** Evaluating *****")
        f1 = evaluate(dev_dataloader)
        print("Epoch={},  f1={}.".format(i+1, f1))
        with open(args.log_file, 'a') as aw:
            aw.write("Epoch={}, now f1 score={:.4f}.".format(i + 1, f1))
            # update optimal parameters
            if f1 > best_score:
                best_score = f1
                model_to_save = model.module if hasattr(model, 'module') else model
                torch.save(model_to_save, args.checkpoint_dir + '/best_model.pth')
                aw.write(" update the best model.\n")
                aw.write(" ---------------------------------------------------- \n")
            else:
                aw.write("keep the old model.\n")
                aw.write(" ---------------------------------------------------- \n")
        model.train()

    print("*" * 30)
    print("Train-{}, Dev-{}".format(len(train_features), len(dev_features)))
    print('Best F1:', best_score)
    print()

    return best_score


# Main
##############################################################################################
if __name__ == '__main__':
    train_features, train_dataloader, dev_features, dev_dataloader = prepare_for_train()
    learn(train_features, train_dataloader, dev_features, dev_dataloader)
