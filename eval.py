import sys, os
import json
import subprocess
from typing import *
from dataloader import read_data
from opt import get_args 

def read_pred_data(path):
    with open(path,'r')as f:
        lines = f.readlines()
        preds = []
        one_sent=[]
        for line in lines:
            if line=='\n':
                one_sent = []
                preds.append(one_sent)
            else:
                one_sent.append(line.strip())
        return preds


def read_label_data(path):
    with open(path,'r')as f:
        lines = f.readlines()
        labels = []
        one_sent = []
        for line in lines:
            if line == '\n':
                one_sent = []
                labels.append(one_sent)
            else:
                one_sent.append(line.strip())
        return labels

def read_raw(raw_path):
    with open(raw_path,'r')as f:
        lines = f.readlines()
        raws = []
        for line in lines:
            raws.append(line.strip().split())
        return raws

def reverse_labeling(tokens, value, slot_name, current_labels):
    if not current_labels:
        current_labels = ['O'] * len(tokens)
    assert len(tokens) == len(current_labels)

    v_tokens = [tk.lower().strip() for tk in value]
    tokens = [tk.lower().strip() for tk in tokens]


    def is_align(i):
        for j in range(len(v_tokens)):
            if not (i + j < len(tokens) and tokens[i + j] == v_tokens[j] and current_labels[i + j] == 'O'):
                return False
        return True

    def fill_label(i):
        current_labels[i] = 'B-' + slot_name
        for j in range(1, len(v_tokens)):
            current_labels[i + j] = 'I-' + slot_name

    for ind, tk in enumerate(tokens):
        if is_align(ind):
            fill_label(ind)

    return current_labels

def inverse(preds,raw,labels):

    sent_num = len(raw)
    all_token_labels = []
    all_tokens = []
    for i in range(sent_num):
        tokens = raw[i]
        token_label = ['O'] * len(tokens)
        for j in range(len(preds[i])):
            for k in preds[i][j]:
                slot_values = k.replace(' .','').split()
                slot_name = labels[i][j]
                token_label = reverse_labeling(tokens, slot_values, slot_name, token_label)
        assert len(token_label) == len(tokens)
        all_token_labels.append(token_label)
        all_tokens.append(tokens)
    return all_tokens,all_token_labels

def load_ground_truth_labels(args,mode):
    train_data, dev_data, test_data = read_data(args)
    ground_truth_labels = []
    ori_tokens = []
    if mode == 'dev':
        for domain_name in dev_data.keys():
            for eid, episode in enumerate(dev_data[domain_name]  ):
                ground_truth_labels.extend(episode['test']['original_bio_seq_out'])
                ori_tokens.extend(episode['test']['original_seq_in'])
    elif mode == 'test':
        for domain_name in test_data.keys():
            for eid, episode in enumerate(test_data[domain_name]  ):
                ground_truth_labels.extend(episode['test']['original_bio_seq_out'])
                ori_tokens.extend(episode['test']['original_seq_in'])
    return ori_tokens,ground_truth_labels



def conll_format_output(target_file, tokens, pred_labels, gold_labels):
    with open(target_file, 'w') as fp:
        for token_lst, pred_label_lst, gold_label_lst in zip(tokens, pred_labels, gold_labels):
            for token, pred_label, true_label in zip(token_lst, pred_label_lst, gold_label_lst):
                fp.write('{0} {1} {2}\n'.format(token, true_label, pred_label))
            fp.write('\n')

def eval_with_script(output_prediction_file, eval_scripts, verbose=True):
    script_args = ['perl', eval_scripts]
    with open(output_prediction_file, 'r') as res_file:
        p = subprocess.Popen(script_args, stdout=subprocess.PIPE, stdin=res_file)
        p.wait()

        std_results = p.stdout.readlines()
        if verbose:
            for r in std_results:
                print(r)
        std_results = str(std_results[1]).split()
    precision = float(std_results[3].replace('%;', ''))
    recall = float(std_results[5].replace('%;', ''))
    f1 = float(std_results[7].replace('%;', '').replace("\\n'", ''))

    return precision, recall, f1

def convert_preds(preds):
    ret = []
    for i in preds:
        one_sent = []
        for j in i:
            one_sent.append(j.split(';'))
        ret.append(one_sent)
    return ret

def eval(result_path,label_path,tar_path,mode,args,raw_path):
    preds = read_pred_data(result_path)

    labels = read_label_data(label_path)

    preds = convert_preds(preds)

    
    ori_tokens, ground_truth_labels = load_ground_truth_labels(args=args,mode=mode)



    print(len(preds))
    print(len(ori_tokens))
    print(len(labels))
    assert len(preds) == len(ori_tokens) == len(labels)

    all_tokens, all_token_labels = inverse(preds, ori_tokens, labels)

 
    target_file = tar_path

    scripts_file = 'conlleval.pl'

    conll_format_output(target_file, all_tokens, all_token_labels, ground_truth_labels)

    precision,recall,f1 = eval_with_script(target_file, scripts_file, True)
    return precision,recall,f1



