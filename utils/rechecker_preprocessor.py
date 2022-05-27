import collections
import random
from re import L
from typing import List, Tuple, Dict
import torch
import pickle
import json
import os
from transformers import AutoTokenizer
from data_loader import FewShotExample, DataItem,FewShotRawDataLoader
from  torch.utils.data import Dataset,DataLoader
def get_labelmap(path):
    with open(path,'r') as f:
        labelmap = json.load(f)
        return labelmap

class Checker_testdata(object):
    def __init__(self,examples:List[FewShotExample]) -> None:
        self.devexamples={}
        self.label2id = {}
        for example in examples:
            example_item = {}
            example_item["domain"] = example.domain_name
            example_item["support"] = {}
            example_item["support"]["original_seq_in"] = []
            example_item["support"]["original_seq_out"] = []
            example_item["support"]["original_bio_seq_out"] = []
            example_item["support"]["prompt_seq_in"]=[]
            example_item["support"]["prompt_seq_out"]=[]
            example_item["support"]["checker_prompt_in"] = []
            example_item["support"]["checker_prompt_out"] = []
            for support_data_item in example.support_data_items:
                example_item["support"]["original_seq_in"].append(support_data_item.seq_in)
                example_item["support"]["original_seq_out"].append(support_data_item.seq_out)
                example_item["support"]["original_bio_seq_out"].append(support_data_item.bio_seq_out)
            example_item["support"]["prompt_seq_in"] = []
            example_item["support"]["prompt_seq_out"] = []
            example_item["test"] = {}
            example_item["test"]["original_seq_in"] = []
            example_item["test"]["original_seq_out"] = []
            example_item["test"]["original_bio_seq_out"] = []
            example_item["test"]["prompt_seq_in"] = []
            example_item["test"]["prompt_seq_out"] = []
            example_item["test"]["checker_prompt_in"] = []
            example_item["test"]["checker_prompt_out"] = []
            for test_data_item in example.test_data_items:
                example_item["test"]["original_seq_in"].append(test_data_item.seq_in)
                example_item["test"]["original_seq_out"].append(test_data_item.seq_out)
                example_item["test"]["original_bio_seq_out"].append(test_data_item.bio_seq_out)
            if example.domain_name not in self.devexamples:
                self.devexamples[example.domain_name] = []
                self.devexamples[example.domain_name].append(example_item)
            elif example.domain_name in self.devexamples:
                self.devexamples[example.domain_name].append(example_item)

    def make_dict(self):
        all_seq_out = []
        all_domain = []
        for dname,domain in self.devexamples.items():
            for example in domain:
                all_seq_out.extend(example["support"]["original_seq_out"])
                for _ in range(len(example["support"]["original_seq_out"])):
                    all_domain.append(example["domain"])
                all_seq_out.extend(example["test"]["original_seq_out"])
                for _ in range(len(example["test"]["original_seq_out"])):
                    all_domain.append(example["domain"])
        
        def purify(l):
            """ remove B- and I- """
            return set([item.replace('B-', '').replace('I-', '') for item in l])
        domain_all_labels = {}
        domain_label2id = {}
        
        for idx in range(len(all_seq_out)):
            if all_domain[idx] not in domain_all_labels:
                domain_all_labels[all_domain[idx]]=[]
                domain_all_labels[all_domain[idx]].extend(all_seq_out[idx])
            else:
                domain_all_labels[all_domain[idx]].extend(all_seq_out[idx])
        for domain,all_labels in domain_all_labels.items():
            label2id = {}
            labels = sorted(list(purify(set(all_labels))))
            for label in labels:
                label2id[label] = len(label2id)
            domain_label2id[domain]=label2id
        self.label2id = domain_label2id
        return domain_label2id

    def add_prompt(self,label2id,labelmap:Dict):
        for domain_name,domain in self.devexamples.items():
            for example in domain:
                
                for seq_in,seq_out,bio_seq_out in zip(example["support"]["original_seq_in"],example["support"]["original_seq_out"],example["support"]["original_bio_seq_out"]):
                    prompt_ins = []
                    prompt_outs = []
                    checker_prompt_ins = []
                    checker_prompt_outs= []

                    #sample check pair
                    label_entity_pair={}
                    
                    
                    for token,label,bio_label in zip(seq_in,seq_out,bio_seq_out):
                        if bio_label.startswith('B-'):
                            if label not in label_entity_pair:
                                label_entity_pair[label]=[token]
                            else:
                                label_entity_pair[label].append(token)
                        if bio_label.startswith('I-'):
                            label_entity_pair[label][-1]+=' '+token
                    
                    all = list(label_entity_pair.items())
                    if len(all)!=0:
                        unsampled=random.sample(all,1)
                    else:
                        unsampled=[]
                    sampled = [i for i in all if i not in unsampled]
                    all = dict(all)
                    sampled = dict(sampled)
                    unsampled = dict(unsampled)

                    

                    checker_prompt_in =  ["'"] + seq_in + ["'"] 
                    checker_prompt_out = ["'"] + seq_in + ["'"] 
                    for label,id in list(label2id[domain_name].items()):
                        if label!='O' and label in sampled:
                            mapped_label = labelmap[domain_name][label]
                            checker_prompt_in += mapped_label.split() + ["refers"] + [ "to"]
                            checker_prompt_out += mapped_label.split() + ["refers"] + [ "to"] 
                            for slot_value in label_entity_pair[label]:
                                checker_prompt_in += slot_value.split() + [";"]
                                checker_prompt_out += slot_value.split() + [";"]
                                
                            if checker_prompt_in[-1] == ';':
                                checker_prompt_in = checker_prompt_in[:-1]
                                checker_prompt_out = checker_prompt_out[:-1]
                                checker_prompt_in += ["."]
                                checker_prompt_out += ["."]
                    
                    for label,id in list(label2id[domain_name].items()):
                        one_checker_prompt_in = [i for i in checker_prompt_in]
                        one_checker_prompt_out = [i for i in checker_prompt_out]
                        if label!='O' and label in unsampled:
                            mapped_label = labelmap[domain_name][label] 
                            one_checker_prompt_in += mapped_label.split()+ ["refers"] + [ "to"] 
                            one_checker_prompt_out += mapped_label.split()+["refers"] + [ "to"]
                            for slot_value in label_entity_pair[label]:
                                one_checker_prompt_out += slot_value.split() + [";"]
                            if one_checker_prompt_out[-1]==';':
                                one_checker_prompt_out = one_checker_prompt_out[:-1]
                            
                            checker_prompt_ins.append(one_checker_prompt_in)
                            checker_prompt_outs.append(one_checker_prompt_out)
                        if label!='O' and label not in all:
                            mapped_label = labelmap[domain_name][label] 
                            one_checker_prompt_in += mapped_label.split()+ ["refers"] + [ "to"] 
                            one_checker_prompt_out += mapped_label.split()+["refers"] + [ "to"] + ['none']
                            checker_prompt_ins.append(one_checker_prompt_in)
                            checker_prompt_outs.append(one_checker_prompt_out)
                    
                    #construct first round prompt
                    for label, id in list(label2id[domain_name].items()):
                        if label!='O':
                            prompt_in =  ["'"] + seq_in + ["'"]
                            prompt_out = ["'"] + seq_in + ["'"]
                            if label in seq_out:
                                mapped_label = labelmap[domain_name][label]
                                prompt_in +=  mapped_label.split() + ["refers"] + [ "to"]
                                prompt_out += mapped_label.split() + ["refers"] + [ "to"]
                                label_mask = [1 if l == label else 0 for l in seq_out]

                                for idx, mask in enumerate(label_mask):
                                    if mask == 1:
                                        if bio_seq_out[idx].startswith('B-'):
                                            label_mask[idx] = 1
                                        if bio_seq_out[idx].startswith('I-'):
                                            label_mask[idx] = -1
                                for idx, mask in enumerate(label_mask):
                                    if mask == 1:
                                        prompt_out = prompt_out + [seq_in[idx]]
                                    if mask == -1:
                                        prompt_out = prompt_out + [seq_in[idx]]
                                    if idx >= 1:
                                        if label_mask[idx - 1] == -1 and mask == 0:
                                            prompt_out = prompt_out + [';']
                                if prompt_out[-1] == ';':
                                    prompt_out = prompt_out[:-1]
                            else:
                                mapped_label = labelmap[domain_name][label]
                                prompt_in += mapped_label.split() + ["refers"] + ["to"]
                                prompt_out +=  mapped_label.split() + ["refers"] + ["to"] + ["none"]
                            prompt_outs.append(prompt_out)
                            prompt_ins.append(prompt_in)
                    example["support"]["prompt_seq_in"].append(prompt_ins)
                    example["support"]["prompt_seq_out"].append(prompt_outs)
                    example["support"]["checker_prompt_in"].append(checker_prompt_ins)
                    example["support"]["checker_prompt_out"].append(checker_prompt_outs)
                    

                for seq_in,seq_out,bio_seq_out in zip(example["test"]["original_seq_in"],example["test"]["original_seq_out"],example["test"]["original_bio_seq_out"]):
                    prompt_ins = []
                    prompt_outs = []
                    checker_prompt_ins = []
                    checker_prompt_outs= []

                    #sample check pair
                    label_entity_pair={}
                    
                    
                    for token,label,bio_label in zip(seq_in,seq_out,bio_seq_out):
                        if bio_label.startswith('B-'):
                            if label not in label_entity_pair:
                                label_entity_pair[label]=[token]
                            else:
                                label_entity_pair[label].append(token)
                        if bio_label.startswith('I-'):
                            label_entity_pair[label][-1]+=' '+token
                    
                    all = list(label_entity_pair.items())
                    if len(all)!=0:
                        unsampled=random.sample(all,1)
                    else:
                        unsampled=[]
                    sampled = [i for i in all if i not in unsampled]
                    all = dict(all)
                    sampled = dict(sampled)
                    unsampled = dict(unsampled)

                    
                    checker_prompt_in =  ["'"] + seq_in + ["'"] 
                    checker_prompt_out = ["'"] + seq_in + ["'"] 
                    for label,id in list(label2id[domain_name].items()):
                        if label!='O' and label in sampled:
                            mapped_label = labelmap[domain_name][label]
                            checker_prompt_in += mapped_label.split() + ["refers"] + [ "to"]
                            checker_prompt_out += mapped_label.split() + ["refers"] + [ "to"] 
                            for slot_value in label_entity_pair[label]:
                                checker_prompt_in += slot_value.split() + [";"]
                                checker_prompt_out += slot_value.split() + [";"]
                            if checker_prompt_in[-1] == ';':
                                checker_prompt_in = checker_prompt_in[:-1]
                                checker_prompt_out = checker_prompt_out[:-1]
                                checker_prompt_in += ["."]
                                checker_prompt_out += ["."]
                                
                    for label,id in list(label2id[domain_name].items()):
                        one_checker_prompt_in = [i for i in checker_prompt_in]
                        one_checker_prompt_out = [i for i in checker_prompt_out]
                        if label!='O' and label in unsampled:
                            mapped_label = labelmap[domain_name][label] 
                            one_checker_prompt_in += mapped_label.split()+ ["refers"] + [ "to"] 
                            one_checker_prompt_out += mapped_label.split()+["refers"] + [ "to"]
                            for slot_value in label_entity_pair[label]:
                                one_checker_prompt_out += slot_value.split() + [";"]
                            if one_checker_prompt_out[-1]==';':
                                one_checker_prompt_out = one_checker_prompt_out[:-1]
                            checker_prompt_ins.append(one_checker_prompt_in)
                            checker_prompt_outs.append(one_checker_prompt_out)
                        if label!='O' and label not in all:
                            mapped_label = labelmap[domain_name][label] 
                            one_checker_prompt_in += mapped_label.split()+ ["refers"] + [ "to"] 
                            one_checker_prompt_out += mapped_label.split()+["refers"] + [ "to"] + ['none']
                            checker_prompt_ins.append(one_checker_prompt_in)
                            checker_prompt_outs.append(one_checker_prompt_out)
      
                    #construct first round prompt
                    for label, id in list(label2id[domain_name].items()):
                        if label!='O':
                            prompt_in =  ["'"] + seq_in + ["'"]
                            prompt_out = ["'"] + seq_in + ["'"]
                            if label in seq_out:
                                mapped_label = labelmap[domain_name][label]
                                prompt_in +=  mapped_label.split() + ["refers"] + [ "to"]
                                prompt_out += mapped_label.split() + ["refers"] + [ "to"]
                                label_mask = [1 if l == label else 0 for l in seq_out]

                                for idx, mask in enumerate(label_mask):
                                    if mask == 1:
                                        if bio_seq_out[idx].startswith('B-'):
                                            label_mask[idx] = 1
                                        if bio_seq_out[idx].startswith('I-'):
                                            label_mask[idx] = -1
                                for idx, mask in enumerate(label_mask):
                                    if mask == 1:
                                        prompt_out = prompt_out + [seq_in[idx]]
                                    if mask == -1:
                                        prompt_out = prompt_out + [seq_in[idx]]
                                    if idx >= 1:
                                        if label_mask[idx - 1] == -1 and mask == 0:
                                            prompt_out = prompt_out + [';']
                                if prompt_out[-1] == ';':
                                    prompt_out = prompt_out[:-1]
                            else:
                                mapped_label = labelmap[domain_name][label]
                                prompt_in += mapped_label.split() + ["refers"] + ["to"]
                                prompt_out +=  mapped_label.split() + ["refers"] + ["to"] + ["none"]
                            prompt_outs.append(prompt_out)
                            prompt_ins.append(prompt_in)
                    example["test"]["prompt_seq_in"].append(prompt_ins)
                    example["test"]["prompt_seq_out"].append(prompt_outs)
                    example["test"]["checker_prompt_in"].append(checker_prompt_ins)
                    example["test"]["checker_prompt_out"].append(checker_prompt_outs)
                    







if __name__=='__main__':
    data_loader = FewShotRawDataLoader()
    test_path = '../original_data/MIT_M/mit_m.10_shot.json'
    labelmap_path = '../original_data/MIT_M/label_verb'
    labelmap = get_labelmap(labelmap_path)  
    test_examples, test_max_len, test_max_support_size = data_loader.load_data(path=test_path)        
    test_data = Checker_testdata(test_examples)
    test_label2id = test_data.make_dict()
    
    test_data.add_prompt(test_label2id,labelmap)
    with open('../prompt_data/MIT_M/mit_m.10_shot.json', 'w',encoding='utf-8') as f:
        json.dump(test_data.devexamples, f)
    
                            
                        
                    
                    
                    

