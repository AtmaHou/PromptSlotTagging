import collections
from typing import List, Tuple, Dict
import torch
import pickle
import json
import os
from transformers import AutoTokenizer
from data_loader import FewShotExample, DataItem,FewShotRawDataLoader
from  torch.utils.data import Dataset,DataLoader
def flatten(l):
    """ convert list of list to list"""
    return [item for sublist in l for item in sublist]

def get_labelmap(path):
    with open(path,'r') as f:
        labelmap = json.load(f)
        return labelmap

def make_dict(examples: List[FewShotExample]) -> (Dict[str, int], Dict[int, str]):
    def purify(l):
        """ remove B- and I- """
        return set([item.replace('B-', '').replace('I-', '') for item in l])



    ''' collect all label from: all test set & all support set '''
    all_labels = []
    label2id = {}
    for example in examples:
        all_labels.extend([data_item.seq_out for data_item in example.test_data_items])
        all_labels.extend([data_item.seq_out for data_item in example.support_data_items])
    ''' collect label word set '''
    label_set = sorted(list(purify(set(flatten(all_labels)))))  # sort to make embedding id fixed
    #print(label_set)

    ''' build dict '''
    label2id['[PAD]'] = len(label2id)  # '[PAD]' in first position and id is 0
    for label in label_set:
        label2id[label] = len(label2id)
    ''' reverse the label2id '''
    id2label = dict([(idx, label) for label, idx in label2id.items()])
    return label2id, id2label




'''def label2onehot(label: str, label2id: dict):
    onehot = [0 for _ in range(len(label2id))]
    onehot[label2id[label]] = 1
    return onehot
def item2label_ids(f_item: FeatureItem, label2id: dict):
    return [label2id[lb] for lb in f_item.labels]

def item2label_onehot( f_item: FeatureItem, label2id: dict):
    return [label2onehot(lb, label2id) for lb in f_item.labels]
'''

class Traindata(object):
    def __init__(self,examples:List[FewShotExample]):
        self.all_seq_in = []
        self.all_seq_out = []
        self.all_bio_seq_out = []
        self.all_domain = []
        for example in examples:
            for test_data_item in example.test_data_items:
                self.all_domain.append(test_data_item.domain)
                self.all_seq_in.append(test_data_item.seq_in)
                self.all_seq_out.append(test_data_item.seq_out)
                self.all_bio_seq_out.append(test_data_item.bio_seq_out)
            for support_data_item in example.support_data_items:
                self.all_domain.append(support_data_item.domain)
                self.all_seq_in.append(support_data_item.seq_in)
                self.all_seq_out.append(support_data_item.seq_out)
                self.all_bio_seq_out.append(support_data_item.bio_seq_out)
    def make_domian_dict(self,all_seq_out,all_domain):
        def purify(l):
            """ remove B- and I- """
            return set([item.replace('B-', '').replace('I-', '') for item in l])
        domain_all_labels = {}
        domain_label2id = {}
        #print(len(all_seq_out))
        for idx in range(len(all_seq_out)):
            if all_domain[idx] not in domain_all_labels:
                domain_all_labels[all_domain[idx]]=[]
                domain_all_labels[all_domain[idx]].extend(all_seq_out[idx])
            else:
                domain_all_labels[all_domain[idx]].extend(all_seq_out[idx])
        for domain,all_labels in domain_all_labels.items():
            label2id = {}
            labels = list(purify(set(all_labels)))
            labels.remove('O')
            labels=sorted(labels)
            for label in labels:
                label2id[label] = len(label2id)
            domain_label2id[domain]=label2id
        return domain_label2id

    # def add_prompt(self,seq_in:List[str],domain_name:str,label2id:Dict,labelmap:Dict)->List[str]:
    #     all_prompt_tokens = []
    #     for label,id in list(label2id[domain_name].items())[1:]:
    #         mapped_label = labelmap[domain_name][label]
    #         prompt_tokens = + ["'"] + seq_in + ["'"] + mapped_label.split() + ["refers"] + ["to"]
    #         all_prompt_tokens.append(prompt_tokens)
    #     return all_prompt_tokens

    def add_prompt_labels(self,seq_in:List[str],domain_name:str,seq_out:List[str],bio_seq_out:List[str],label2id:Dict,labelmap:Dict):
        all_prompt_ins = []
        all_prompt_outs = []
        for label, id in list(label2id[domain_name].items()):
            if label!='O':
                if label in seq_out:
                    mapped_label = labelmap[domain_name][label]
                    prompt_in =  ["'"] + seq_in + ["'"] + mapped_label.split() + ["refers"] + ["to"]
                    prompt_out =  ["'"] + seq_in + ["'"] + mapped_label.split() + ["refers"] + ["to"]
                    label_mask = [1 if l==label else 0 for l in seq_out]

                    for idx,mask in enumerate(label_mask):
                        if mask==1:
                            if bio_seq_out[idx].startswith('B-'):
                                label_mask[idx]=1
                            if bio_seq_out[idx].startswith('I-'):
                                label_mask[idx]=-1
                    for idx,mask in enumerate(label_mask):
                        if mask==1:
                            prompt_out = prompt_out + [seq_in[idx]]
                        if mask==-1:
                            prompt_out = prompt_out + [seq_in[idx]]
                        if idx>=1:
                            if label_mask[idx-1]==-1 and mask==0:
                                prompt_out = prompt_out + [';']
                    if prompt_out[-1] == ';':
                        prompt_out=prompt_out[:-1]
                    #for _,_ in list(label2id[domain_name].items())[1:]:
                    all_prompt_ins.append(prompt_in)
                    all_prompt_outs.append(prompt_out)
                else:
                    '''p = torch.rand(1).item()
                    if p>0.8:
                        mapped_label = labelmap[domain_name][label]
                        prompt_in =  ["'"] + seq_in + ["'"] + mapped_label.split() + ["refers"] + ["to"]
                        prompt_out =  ["'"] + seq_in + ["'"] + mapped_label.split() + ["refers"] + ["to"] + ["none"]
                    else:
                        prompt_in = None
                        prompt_out = None'''
                    mapped_label = labelmap[domain_name][label]
                    prompt_in =  ["'"] + seq_in + ["'"] + mapped_label.split() + ["refers"] + ["to"]
                    prompt_out =  ["'"] + seq_in + ["'"] + mapped_label.split() + ["refers"] + [
                        "to"] + ["none"]
                    all_prompt_ins.append(prompt_in)
                    all_prompt_outs.append(prompt_out)
            '''if prompt_in:
                all_prompt_ins.append(prompt_in)
            if prompt_out:
                all_prompt_outs.append(prompt_out)'''
        return  all_prompt_ins,all_prompt_outs

if __name__ == "__main__":

    data_loader = FewShotRawDataLoader()
    train_path = '../original_data/snips/xval_snips/snips_train_1.json'

    labelmap_path ='../original_data/snips/xval_snips/label_verb'
    
    labelmap = get_labelmap(labelmap_path)
    train_examples, train_max_len, train_max_support_size = data_loader.load_data(path=train_path)
    
    train_label2id, train_id2label = make_dict(train_examples)
    
    train_data = Traindata(train_examples)
    train_label2id = train_data.make_domian_dict(train_data.all_seq_out,train_data.all_domain)
    
    all_prompt_out=[]
    all_prompt_in=[]
    for idx in range(len(train_data.all_seq_in)):
        
        prompt_in,prompt_out = train_data.add_prompt_labels(train_data.all_seq_in[idx], train_data.all_domain[idx],
                                                  train_data.all_seq_out[idx], train_data.all_bio_seq_out[idx],
                                                  train_label2id, labelmap)
        
        all_prompt_out.append(prompt_out)
        all_prompt_in.append(prompt_in)
    all_write_train_data = []
    for idx in range(len(train_data.all_seq_in)):
        write_train_data = {}
        write_train_data["domain"] = train_data.all_domain[idx]
        write_train_data["original_seq_in"] = train_data.all_seq_in[idx]
        write_train_data["original_seq_out"] = train_data.all_seq_out[idx]
        write_train_data["prompt_seq_in"] = all_prompt_in[idx]
        write_train_data["prompt_seq_out"] = all_prompt_out[idx]
        write_train_data["label2id"] = train_label2id[train_data.all_domain[idx]]
        all_write_train_data.append(write_train_data)
    with open('../prompt_data/snips/prompt_snips/snips_train_1.json', 'w',encoding='utf-8') as f:
        json.dump(all_write_train_data, f)
    