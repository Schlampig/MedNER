import os
import random
import argparse
import numpy as np
from time import time
import torch
from torch.utils.data import TensorDataset, DataLoader
import bert_codes.tokenization as tokenization
from train import data2fea, get_entity_dict, compare_entity_dict  # NOTE： it can trigger the utils.check_args() to delete train/dev log files.
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
# set args
parser = argparse.ArgumentParser()
parser.add_argument('--gpu_ids', type=str, default='2, 3')
# training parameter
parser.add_argument('--test_batch_size', type=int, default=256)
parser.add_argument('--float16', type=bool, default=True)  # only sm >= 7.0 (tensorcores)
parser.add_argument('--seed', type=int, default=42)
# data and model dir
parser.add_argument('--test_dir', type=str, default='./datasets/ner_test.json')
parser.add_argument('--feature_test_dir', type=str, default='./datasets/fea_ner_test.json')
parser.add_argument('--vocab_file', type=str, default='./pretrained_models/bert_chinese/vocab.txt')
parser.add_argument('--checkpoint_dir', type=str, default='check_points/base_ner')
parser.add_argument('--predict_file', type=str, default='predict_log.txt')
args = parser.parse_args()

# tokenizer initialization
tokenizer = tokenization.BertTokenizer(vocab_file=args.vocab_file, do_lower_case=True)

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

# load model
print('***** Loading Model *****')
model = torch.load(args.checkpoint_dir + "/best_model.pth")
model.to(device)

print("Configuration Time: {}".format(time() - t_config))


# Predict Batch
##############################################################################################
def prepare_for_test():
    global args
    # get features
    test_features = data2fea(load_path=args.test_dir, save_path=args.feature_test_dir)
    # get test dataloader
    test_input_ids = torch.tensor([f['input_ids'] for f in test_features], dtype=torch.long)
    test_input_mask = torch.tensor([f['input_mask'] for f in test_features], dtype=torch.long)
    test_input_segments = torch.tensor([f['input_segments'] for f in test_features], dtype=torch.long)
    test_input_tags = torch.tensor([f['input_tags'] for f in test_features], dtype=torch.long)
    test_tensor = TensorDataset(test_input_ids, test_input_mask, test_input_segments, test_input_tags)
    test_dataloader = DataLoader(test_tensor, batch_size=args.test_batch_size, shuffle=True)
    print("Test-{}".format(len(test_features)))
    return test_features, test_dataloader


def print_and_save_batch_dict(input_id, d_true, d_pred):
    global args, tokenizer
    lst_token = tokenizer.convert_ids_to_tokens(input_id.cpu().numpy())
    lst_token = [t.replace("##", "") if t.startswith("##") else t for t in lst_token]
    s_token = "".join(lst_token).replace("[PAD]", "").replace("[CLS]", "").replace("[SEP]", "")
    with open(args.predict_file, 'w') as aw_dev:
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
    with open(args.predict_file, 'w') as f:
        f.write("True: {} \n".format(s_true))
        f.write("Pred: {} \n".format(s_pred))
        f.write("\n")
    return None


def predict():
    global args, model, DICT_LABEL
    print("***** Preprocessing *****")
    _, test_dataloader = prepare_for_test()

    print("***** Predict *****")
    model.eval()
    d_res = {k: {"precision": [], "recall": [], "f1": []} for k in DICT_LABEL.keys()}
    with torch.no_grad():
        for step, batch in enumerate(test_dataloader):
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
            with open(args.predict_file, 'w') as f:
                f.write("Result of batch {} is: \n {} \n".format(step, res_batch))
                f.write(" ---------------------------------------------------- \n")
            print("Result of batch {} is: \n {} \n ".format(step, res_batch))
        # get final scores
        f1 = np.mean([np.mean(v["f1"]) for k, v in d_res.items()])
        with open(args.predict_file, 'w') as f:
            f.write("F1 of all batches is: {:.4f} \n".format(f1))
            f.write(" ---------------------------------------------------- \n")
        print("F1 of all batches is: {:.4f} \n".format(f1))
    return f1


# Main
##############################################################################################
if __name__ == '__main__':
    res = predict()

