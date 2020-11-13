import os
import re
import collections
import torch
from nltk.translate.bleu_score import sentence_bleu
import jieba
SPIECE_UNDERLINE = '▁'


# Model or File Utils
##############################################################################################
def check_args(args, rank=0):
    args.setting_file = os.path.join(args.checkpoint_dir, args.setting_file)
    args.log_file = os.path.join(args.checkpoint_dir, args.log_file)
    args.log_dev_file = os.path.join(args.checkpoint_dir, args.log_dev_file)  # store results of each batch in evaluation phrase
    if os.path.exists(args.setting_file) and rank == 0:
        os.remove(args.setting_file)
    if os.path.exists(args.log_file) and rank == 0:
        os.remove(args.log_file)
    if os.path.exists(args.log_dev_file) and rank == 0:  # settings for log_dev_file
        os.remove(args.log_dev_file)
    if rank == 0:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        with open(args.setting_file, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            print('------------ Options -------------')
            for k in args.__dict__:
                v = args.__dict__[k]
                opt_file.write('%s: %s\n' % (str(k), str(v)))
                print('%s: %s' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
            print('------------ End -------------')

            
def torch_init_model(model, init_checkpoint):
    state_dict = torch.load(init_checkpoint, map_location='cpu')
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})

        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix='' if hasattr(model, 'bert') else 'bert.')

    print("missing keys:{}".format(missing_keys))
    print('unexpected keys:{}'.format(unexpected_keys))
    print('error msgs:{}'.format(error_msgs))


# Pre-processing Utils
##############################################################################################
def kmp_match(s1, s2):
    def gen_next(s):
        k = -1
        n = len(s)
        m = 0
        lst = [0] * n
        lst[0] = -1
        while m < n-1:
            if k == -1 or s[k] == s[m]:
                k += 1
                m += 1
                lst[m] = k
            else:
                k = lst[k]
        return lst
    next_list = gen_next(s2)
    ans = -1
    i = 0
    j = 0
    while i < len(s1):
        if s1[i] == s2[j] or j == -1:
            i += 1
            j += 1
        else:
            j = next_list[j]
        if j == len(s2):
            ans = i - len(s2)
            break
    return ans


def _is_chinese_char(cp):
    if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
            (cp >= 0x3400 and cp <= 0x4DBF) or  #
            (cp >= 0x20000 and cp <= 0x2A6DF) or  #
            (cp >= 0x2A700 and cp <= 0x2B73F) or  #
            (cp >= 0x2B740 and cp <= 0x2B81F) or  #
            (cp >= 0x2B820 and cp <= 0x2CEAF) or
            (cp >= 0xF900 and cp <= 0xFAFF) or  #
            (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
        return True
    return False


# 判断是否是符号
def is_fuhao(c):
    if c in ('。', '，', ',', '！', '!', '？', '?', '；', ';', '、', '：', ':', '（', '(', '）', ')', '－', '~', '～',
             '「', '」', '《', '》', '"', '“', '”', '$', '『', '』', '—', '-', '‘', '’', '\'', '[', '【',
             ']', '】', '=', '|', '<', '>'):
        return True
    return False


def _tokenize_chinese_chars(text):
    output = []
    for char in text:
        cp = ord(char)
        if _is_chinese_char(cp) or is_fuhao(char):
            if len(output) > 0 and output[-1] != SPIECE_UNDERLINE:
                output.append(SPIECE_UNDERLINE)
            output.append(char)
            output.append(SPIECE_UNDERLINE)
        else:
            output.append(char)
    return "".join(output)


def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F or c == SPIECE_UNDERLINE:
        return True
    return False


def split_sent(text):
    # split text into tokens
    context_chs = _tokenize_chinese_chars(text)
    prev_is_whitespace = True
    doc_tokens = []
    for c in context_chs:
        if is_whitespace(c):
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                doc_tokens.append(c)
            else:
                doc_tokens[-1] += c
            prev_is_whitespace = False
    return doc_tokens


def combine_tokens(tokens):
    output_text = []
    for tok in tokens:
        if len(tok) == 1 and _is_chinese_char(ord(tok)):
            output_text.append(tok)
        elif tok.startswith('##'):
            output_text.append(tok.replace('##', ''))
        elif len(output_text) > 0:
            if len(output_text[-1]) == 1 and (_is_chinese_char(ord(output_text[-1])) or is_fuhao(output_text[-1])):
                output_text.append(tok)
            else:
                output_text.append(' ' + tok)
        else:
            output_text.append(tok)

    output_text = "".join(output_text).strip()
    return output_text


