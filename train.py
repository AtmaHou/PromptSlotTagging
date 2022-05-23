import argparse
import os
import random
import logging
import json
import math
from tqdm import tqdm, trange
import copy
import numpy as np
import torch
import torch.nn.functional as F
import time

from eval import eval
from dataloader import CheckerDevSupportEpisode, CheckerDevTestEpisode, TrainDataset, traincollate, read_data, DevDataset, DevSupportEpisode, DevTestEpisode, devcollate, checker_devcollate, test_checker_collate
from torch.utils.data import (DataLoader, RandomSampler, TensorDataset)
from transformers import (GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, AdamW, WEIGHTS_NAME, CONFIG_NAME,
                          get_linear_schedule_with_warmup)
import pickle
from opt import get_args
from model import GPT2GenModel

def generate(model,tokenizer,batch_prompt_in,encoded_raw_sent,each_in_length,num_samples = 1,length = 1,temperature = 1,top_k=1,top_p=0.0):

    encoded_raw_sent = torch.tensor(encoded_raw_sent,device='cuda' if torch.cuda.is_available() else 'cpu')
    original_each_in_length = [i for i in each_in_length]

    generated = batch_prompt_in
    
    with torch.no_grad():
        end_flags=torch.tensor([1]*batch_prompt_in.shape[0],device='cuda' if torch.cuda.is_available() else 'cpu')
        for current_length in range(length):
            inputs = {'input_ids': generated}

            outputs = model(
                **inputs)
            logits = outputs[1]

            t = []
            for i, j in enumerate(each_in_length):
                
                t.append(logits[i].narrow(dim=0, start=j - 1, length=1))
            next_token_logits = torch.cat(t, dim=0)
            next_token_logits = torch.gather(next_token_logits, 1, encoded_raw_sent)
            next_token = torch.argmax(next_token_logits, dim=-1)
            next_token.unsqueeze_(-1)
            next_token = torch.gather(encoded_raw_sent, 1, next_token)
            torch_next_token = next_token.squeeze(1)
            next_token=torch_next_token.tolist()
            
            #collect '.',stop generating when all sentences in the batch meet '.'
            this_end = torch.where(torch_next_token==764,0,1)
            end_flags*=this_end

            g = []
            for i, j in enumerate(each_in_length):
                p1 = generated[i].narrow(dim=0, start=0, length=j).unsqueeze(0)
                p2 = torch.tensor([next_token[i]], device='cuda' if torch.cuda.is_available() else 'cpu').unsqueeze(0)
                p3 = generated[i].narrow(dim=0, start=j, length=generated.shape[-1] - j).unsqueeze(0)
                g.append(torch.cat([p1, p2, p3], dim=1))
                each_in_length[i] += 1
            generated = torch.cat(g, dim=0)
            if torch.sum(end_flags).item()==0:
                break

    #split generated slots and input sentence
    generated = generated.cpu().tolist()
    texts = []
    slots = []
    for i, j in enumerate(generated):
        if 50256 in j:
            idx = j.index(50256)
            text = j[:idx]
        else:
            text = j
        texts.append(tokenizer.decode(text, clean_up_tokenization_spaces=False))
        slot = text[original_each_in_length[i]:]
        slots.append(tokenizer.decode(slot, clean_up_tokenization_spaces=False))
    return texts, slots
    


def finetune_on_support(args,test_model,optimizer,scheduler,dataloader,logger):
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        test_model, optimizer1 = amp.initialize(test_model, optimizer, opt_level=args.fp16_opt_level)

    test_model.train()
    for idx_epochs in range(int(args.num_finetune_epochs)):
        
        all_loss = 0.0
        nb_steps = 0

        
        optimizer.zero_grad()

        total_step, update_step = 0, 0
        for id, data in enumerate(dataloader):
            batch_prompt_in, batch_prompt_out, batch_masked_prompt_out, batch_attention_mask, batch_raw_sent, batch_in_length = data
            outputs = test_model(batch_prompt_out, labels=batch_masked_prompt_out,
                                attention_mask=batch_attention_mask)
            loss = outputs[0]
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            all_loss += loss.mean().item()
            nb_steps += 1

            if (total_step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(test_model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                update_step += 1

            
            total_step += 1
        all_loss = all_loss / nb_steps
        

        logger.info(
            "Epoch: {}, Step: {}, Finetuning stage 1 loss: {:.2e} update_step: {:.2e}\n".format(
                idx_epochs, total_step, all_loss, update_step))


def generate_procedure(args,test_model,test_model_2,tokenizer,test_test_dataloader):
    #stage 1 gen
    last_sent = ''
    all_raw_lines,all_label_lines,all_results_lines = [],[],[]
    domain_name =''
    for id, data in enumerate(test_test_dataloader):
        batch_prompt_in, batch_prompt_out, batch_masked_prompt_out, batch_attention_mask, batch_raw_sent, batch_in_length = data
        outs, slots = generate(model=test_model, tokenizer=tokenizer, batch_prompt_in=batch_prompt_in,
                                encoded_raw_sent=batch_raw_sent, each_in_length=batch_in_length, length=args.gen_length)
        count =0
        with open(args.test_path+'verb_label', 'r') as f:
            label_map = json.load(f)
        
        for out, slot in zip(outs, slots):
            if args.dataset=='mit':
                domain_name="atis"
            else:
                domain_name = out.split('"')[0].split(':')[0].strip()
            label_num = len(list(label_map[domain_name].items()))

            if out.split('"')[1] != last_sent:
                count=1
            else:
                count+=1
            if out.split('"')[1] != last_sent or count > label_num and count % label_num==1:
                all_label_lines.append('\n')
                
                
            label = label_map[domain_name][out.split('"')[-1].split('refers to')[0].strip()]
            all_label_lines.append(label)

            if out.split('"')[1] != last_sent or count > label_num and count % label_num==1:
                raw = out.split('"')[1].strip()
                all_raw_lines.append(raw)
                
            if out.split('"')[1] != last_sent or count > label_num and count % label_num==1:
                all_results_lines.append('\n')
                last_sent = out.split('"')[1]
            
            if not slot.endswith('.'):
                    slot+=' .'
            all_results_lines.append(slot)
            
    
    labels = []
    one_sent_label = []
    for line in all_label_lines:
        if line == '\n':
            one_sent_label = []
            labels.append(one_sent_label)
        else:
            one_sent_label.append(line.strip())
    

    results = []
    one_sent_results=[]
    for line in all_results_lines:
        if line=='\n':
            one_sent_results = []
            results.append(one_sent_results)
        else:
            one_sent_results.append(line.strip())   
    

    raws = [i.strip() for i in all_raw_lines]

    #write stage1 generation result
    with open(args.pred_path+args.test_file+'/raw1.txt', 'a') as f:
        for i in raws:
            f.write(i)
            f.write('\n')

    with open(args.pred_path+args.test_file+'/label1.txt', 'a') as f:
        for i in labels:
            f.write('\n')
            for j in i:
                f.write(j)
                f.write('\n')

    with open(args.pred_path+args.test_file+'/result1.txt', 'a') as f:
        for i in results:
            f.write('\n')
            for j in i:
                f.write(j)
                f.write('\n')


    #stage2 gen
    with open(args.test_path+'label_verb','r')as f:
        label2verb=json.load(f)
    
    with open(args.test_path+'verb_label', 'r') as f:
            label_map = json.load(f)
    
    #construct stage2 generation inputs based on results from stage1
    checker_test_test_dataset = CheckerDevTestEpisode(args.dataset,raws,labels,results,label2verb[domain_name],domain_name,tokenizer)
    checker_test_test_dataloader = DataLoader(dataset=checker_test_test_dataset,batch_size=args.gen_batch_size,collate_fn=test_checker_collate)
    
    
    last_sent = ''
    all_label_lines_2,all_raw_lines_2,all_results_lines_2 = [],[], []
    for id, data in enumerate(checker_test_test_dataloader):
        batch_prompt_in , batch_raw_sent, batch_in_length = data
        outs, slots = generate(model=test_model_2, tokenizer=tokenizer, batch_prompt_in=batch_prompt_in,
                                encoded_raw_sent=batch_raw_sent, each_in_length=batch_in_length, length=args.gen_length)
        count =0
        
        for out, slot in zip(outs, slots):
            if args.dataset=='mit':
                domain_name="atis"
            else:
                domain_name = out.split('"')[0].split(':')[0].strip()
            label_num = len(list(label_map[domain_name].items()))

            
            if out.split('"')[1] != last_sent:
                count=1
            else:
                count+=1
            if out.split('"')[1] != last_sent or count > label_num and count % label_num==1:
                all_label_lines_2.append('\n')
            label_name = out.split('"')[-1].split('refers to')[-2].strip()
            if '.'in label_name:
                label_name = label_name.split('.')[-1].strip()
            label_2 = label_map[domain_name][label_name]
            all_label_lines_2.append(label_2)
                
            if out.split('"')[1] != last_sent or count > label_num and count % label_num==1:
                raw_2 = out.split('"')[1].strip()
                all_raw_lines_2.append(raw_2)

            if out.split('"')[1] != last_sent or count > label_num and count % label_num==1:
                all_results_lines_2.append('\n')
                last_sent = out.split('"')[1]
            if not slot.endswith('.'):
                slot+=' .'
            all_results_lines_2.append(slot)
    
    labels_2 = []
    one_sent_label_2 = []
    for line in all_label_lines_2:
        if line == '\n':
            one_sent_label_2 = []
            labels_2.append(one_sent_label_2)
        else:
            one_sent_label_2.append(line.strip())
    

    results_2 = []
    one_sent_results_2=[]
    for line in all_results_lines_2:
        if line=='\n':
            one_sent_results_2 = []
            results_2.append(one_sent_results_2)
        else:
            one_sent_results_2.append(line.strip())   
    

    raws_2 = [i.strip() for i in all_raw_lines_2]

    for id,data in enumerate(zip(labels,results,labels_2,results_2)):
        label1,result1,label2,result2 = data
        unrecognized_label = []
        for l,r in zip(label1,result1):
            if r.strip().startswith('none'):
                unrecognized_label.append(l)
        checked_label = []
        for l,r in zip(label2,result2):
            if not r.strip().startswith('none'):
                checked_label.append(l)
        for change_label in checked_label:
            if change_label in unrecognized_label:
                change_idx_1 = label1.index(change_label)
                change_idx_2 = label2.index(change_label)
                results[id][change_idx_1]=result2[change_idx_2]

    #write stage2 and final results
    with open(args.pred_path+args.test_file+'/raw.txt', 'a') as f:
        for i in raws:
            f.write(i)
            f.write('\n')

    with open(args.pred_path+args.test_file+'/label.txt', 'a') as f:
        for i in labels:
            f.write('\n')
            for j in i:
                f.write(j)
                f.write('\n')

    with open(args.pred_path+args.test_file+'/result.txt', 'a') as f:
        for i in results:
            f.write('\n')
            for j in i:
                f.write(j)
                f.write('\n')


    with open(args.pred_path+args.test_file+'/raw2.txt', 'a') as f:
        for i in raws_2:
            f.write(i)
            f.write('\n')

    with open(args.pred_path+args.test_file+'/label2.txt', 'a') as f:
        for i in labels_2:
            f.write('\n')
            for j in i:
                f.write(j)
                f.write('\n')

    with open(args.pred_path+args.test_file+'/result2.txt', 'a') as f:
        for i in results_2:
            f.write('\n')
            for j in i:
                f.write(j)
                f.write('\n')
    
    return 


def test(args,model,tokenizer,logger,test_data):

    # prepare dataset
    test_dataset = DevDataset(test_data,tokenizer)
    bar = tqdm(test_dataset,desc='Testing')

    # make output dir
    if not os.path.exists(args.pred_path):
        os.mkdir(args.pred_path)
    if not os.path.exists(args.pred_path+args.test_file):
        os.mkdir(args.pred_path+args.test_file)
    
    
    for eid, test_ep in enumerate(bar):
        #copy first-round generation model for each episode
        test_model = copy.deepcopy(model)
        test_model.to(args.device)
        
        #construct all dataloaders
        support_encoded_prompt_in, support_encoded_prompt_out, support_encoded_raw_sent, support_mask_encoded_prompt_out, test_encoded_prompt_in, test_encoded_prompt_out, test_encoded_raw_sent, test_mask_encoded_prompt_out,support_encoded_checker_prompt_in, support_encoded_checker_prompt_out, support_encoded_checker_raw_sent, support_mask_encoded_checker_prompt_out, test_encoded_checker_prompt_in, test_encoded_checker_prompt_out, test_encoded_checker_raw_sent, test_mask_encoded_checker_prompt_out = test_ep
        test_support_episode = DevSupportEpisode(support_encoded_prompt_in, support_encoded_prompt_out,
                                                support_encoded_raw_sent, support_mask_encoded_prompt_out,
                                                tokenizer)
        test_test_episode = DevTestEpisode(test_encoded_prompt_in, test_encoded_prompt_out,
                                          test_encoded_raw_sent, test_mask_encoded_prompt_out, tokenizer)
        checker_test_support_episode = CheckerDevSupportEpisode(support_encoded_checker_prompt_in, support_encoded_checker_prompt_out,
                                          support_encoded_checker_raw_sent, support_mask_encoded_checker_prompt_out, tokenizer)
        test_support_dataloader = DataLoader(dataset=test_support_episode,batch_size=2,shuffle=True,collate_fn=devcollate)
        test_test_dataloader = DataLoader(dataset=test_test_episode,batch_size=args.gen_batch_size,collate_fn=devcollate)
        checker_test_support_dataloader = DataLoader(dataset=checker_test_support_episode,batch_size=2,collate_fn=checker_devcollate)
       

        # prepare first-round optimizer and scheduler
        if args.max_finetune_steps > 0:
            total = args.max_finetune_steps
            args.num_finetune_epochs = args.max_finetune_steps // (
                    len(test_support_dataloader) // args.gradient_accumulation_steps) + 1
        else:
            total = len(test_support_dataloader) // args.gradient_accumulation_steps * args.num_finetune_epochs

        param_optimizer = list(test_model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer1 = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler1 = get_linear_schedule_with_warmup(optimizer1, num_warmup_steps=args.warmup_rate * total,num_training_steps=total)
        
        #first-round finetune on the support set
        finetune_on_support(args=args,test_model=test_model,optimizer=optimizer1,scheduler=scheduler1,dataloader=test_support_dataloader,logger=logger)
        
        #copy second-round generation model for each episode
        test_model_2 = copy.deepcopy(test_model)
        test_model_2.to(args.device)
        test_model_2.train()

        # prepare second-round optimizer and scheduler
        if args.max_finetune_steps > 0:
            total = args.max_finetune_steps
            args.num_finetune_epochs = args.max_finetune_steps // (
                    len(checker_test_support_dataloader) // args.gradient_accumulation_steps) + 1
        else:
            total = len(checker_test_support_dataloader) // args.gradient_accumulation_steps * args.num_finetune_epochs
        param_optimizer = list(test_model_2.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer2 = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler2 = get_linear_schedule_with_warmup(optimizer2, num_warmup_steps=args.warmup_rate * total,num_training_steps=total)

        #second-round finetune on the support set ,model copied from first round finetuned model.
        finetune_on_support(args=args,test_model=test_model_2,optimizer=optimizer2,scheduler=scheduler2,dataloader=checker_test_support_dataloader,logger=logger)
        #whole generation procedure
        generate_procedure(args,test_model,test_model_2,tokenizer,test_test_dataloader)
    _,_,test_score1 = eval(result_path=args.pred_path+args.test_file+'/result1.txt',label_path =args.pred_path+args.test_file+'/label1.txt',tar_path = args.pred_path+args.test_file+'/tar1.txt',raw_path=args.pred_path+args.test_file+'/raw1.txt',mode='test',args=args)
    _,_,test_score2 = eval(result_path=args.pred_path+args.test_file+'/result.txt',label_path =args.pred_path+args.test_file+'/label.txt',tar_path = args.pred_path+args.test_file+'/tar.txt',raw_path=args.pred_path+args.test_file+'/raw.txt',mode='test',args=args)
    return test_score1,test_score2

def model_selection(args,model,cpt_file,tokenizer,logger,best_score,dev_data):

    dev_dataset = DevDataset(dev_data, tokenizer)
    bar = tqdm(dev_dataset, desc="Dev")
    if not os.path.exists(args.model_selection_path):
        os.mkdir(args.model_selection_path)

    if not os.path.exists(args.model_selection_path+str(args.dataset)+str(args.dev_file)+'/'):
        os.mkdir(args.model_selection_path+str(args.dataset)+str(args.dev_file)+'/')
    for eid, dev_ep in enumerate(bar):
        dev_model = copy.deepcopy(model)
        dev_model.to(args.device)
        support_encoded_prompt_in, support_encoded_prompt_out, support_encoded_raw_sent, support_mask_encoded_prompt_out, test_encoded_prompt_in, test_encoded_prompt_out, test_encoded_raw_sent, test_mask_encoded_prompt_out,support_encoded_checker_prompt_in, support_encoded_checker_prompt_out, support_encoded_checker_raw_sent, support_mask_encoded_checker_prompt_out, test_encoded_checker_prompt_in, test_encoded_checker_prompt_out, test_encoded_checker_raw_sent, test_mask_encoded_checker_prompt_out = dev_ep
        dev_support_episode = DevSupportEpisode(support_encoded_prompt_in, support_encoded_prompt_out,
                                                support_encoded_raw_sent, support_mask_encoded_prompt_out,
                                                tokenizer)
        dev_test_episode = DevTestEpisode(test_encoded_prompt_in, test_encoded_prompt_out,
                                          test_encoded_raw_sent, test_mask_encoded_prompt_out, tokenizer)
        dev_support_dataloader = DataLoader(dataset=dev_support_episode, batch_size=2, shuffle=True,
                                            collate_fn=devcollate)
        dev_test_dataloader = DataLoader(dataset=dev_test_episode, batch_size=args.gen_batch_size,
                                         collate_fn=devcollate)

        if args.max_finetune_steps > 0:
            total = args.max_finetune_steps
            args.num_finetune_epochs = args.max_finetune_steps // (
                        len(dev_support_dataloader) // args.gradient_accumulation_steps) + 1
        else:
            total = len(dev_support_dataloader) // args.gradient_accumulation_steps * args.num_finetune_epochs

        param_optimizer = list(dev_model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_rate*total,
                                                    num_training_steps=total)
        if args.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            dev_model, optimizer = amp.initialize(dev_model, optimizer, opt_level=args.fp16_opt_level)

        dev_model.train()

        for idx_epochs in trange(int(args.num_finetune_epochs), desc="Epoch"):
            all_loss = 0.0
            nb_steps = 0

            ft_tdqm_bar = tqdm(dev_support_dataloader, desc="Modelselection_FineTuning")
            optimizer.zero_grad()

            total_step, update_step=0, 0
            for id,data in enumerate(ft_tdqm_bar):
                batch_prompt_in, batch_prompt_out, batch_masked_prompt_out, batch_attention_mask, batch_raw_sent,batch_in_length  = data
                outputs = dev_model(batch_prompt_out, labels=batch_masked_prompt_out,attention_mask=batch_attention_mask)
                loss = outputs[0]
                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                all_loss += loss.mean().item()
                nb_steps += 1

                if (total_step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(dev_model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    update_step += 1

                total_step += 1
            all_loss = all_loss / nb_steps
            ft_tdqm_bar.desc = "ModelSelection_Finetuning loss: {:.2e} lr: {:.2e}".format(all_loss, scheduler.get_lr()[0])

            logger.info(
                "Epoch: {}, Step: {}, Finetuning loss: {:.2e} update_step: {:.2e}\n".format(
                    idx_epochs, total_step, all_loss, update_step))


        gen_tdqm_bar = tqdm(dev_test_dataloader, desc="Generating")
        last_sent = ''
        for id, data in enumerate(gen_tdqm_bar):
            batch_prompt_in, batch_prompt_out, batch_masked_prompt_out, batch_attention_mask, batch_raw_sent, batch_in_length = data

            outs, slots = generate(model=dev_model, tokenizer=tokenizer, batch_prompt_in=batch_prompt_in,
                                   encoded_raw_sent=batch_raw_sent, each_in_length=batch_in_length, length=20)
            count = 0
            with open(args.dev_path+'verb_label', 'r') as f:
                label_map = json.load(f)
            
            for out, slot in zip(outs, slots):
                if args.dataset=='mit':
                    domain_name="atis"
                else:
                    domain_name = out.split('"')[0].split(':')[0].strip()
                label_num = len(list(label_map[domain_name].items()))
                if not os.path.exists(args.model_selection_path+str(args.dataset)+str(args.dev_file)+'/'+cpt_file+'/'):
                    os.mkdir(args.model_selection_path+str(args.dataset)+str(args.dev_file)+'/'+cpt_file+'/')

                with open(args.model_selection_path+str(args.dataset)+str(args.dev_file)+'/'+cpt_file+'/label_'+cpt_file+'.txt', 'a') as f:
                    if out.split('"')[1] != last_sent:
                        count = 1
                    else:
                        count += 1
                    if out.split('"')[1] != last_sent or count > label_num and count % label_num==1:
                        f.write('\n')
                    f.write(label_map[domain_name][out.split('"')[-1].split('refers to')[0].strip()])
                    f.write('\n')
                with open(args.model_selection_path+str(args.dataset)+str(args.dev_file)+'/'+cpt_file+'/raw_'+cpt_file+'.txt', 'a') as f:
                    if out.split('"')[1] != last_sent or count > label_num and count % label_num==1:
                        f.write(out.split('"')[1].strip())
                        f.write(('\n'))
                with open(args.model_selection_path+str(args.dataset)+str(args.dev_file)+'/'+cpt_file+'/result_'+cpt_file+'.txt', 'a') as f:
                    if out.split('"')[1] != last_sent or count > label_num and count % label_num==1:
                        f.write('\n')
                        last_sent = out.split('"')[1]
                    
                    f.write(slot)
                    f.write('\n')

    _,_,dev_score = eval(result_path=args.model_selection_path+str(args.dataset)+str(args.dev_file)+'/'+cpt_file+'/result_'+cpt_file+'.txt',
                        label_path = args.model_selection_path+str(args.dataset)+str(args.dev_file)+'/'+cpt_file+'/label_'+cpt_file+'.txt',
                        tar_path = args.model_selection_path+str(args.dataset)+str(args.dev_file)+'/'+cpt_file+'/tar_'+cpt_file+'.txt',
                        raw_path = args.model_selection_path+str(args.dataset)+str(args.dev_file)+'/'+cpt_file+'/raw_'+cpt_file+'.txt',
                        mode='dev',args=args)
    
    best_model = None

    if dev_score > best_score:
        logger.info(" === Found new best!! "+cpt_file+"=== ")

        ''' store new best model  '''
        best_model = copy.deepcopy(model)  # copy model to avoid writen by latter training
        ''' save model file '''
        
        if os.path.exists(args.ft_model_output_dir+'_'+str(args.dev_file)+'/best'):
            os.system(r'rm -rf ' + args.ft_model_output_dir+'_'+str(args.dev_file)+'/best')
            save_model(finetune=True, args=args, logger=logger, model=best_model, tokenizer=tokenizer,train_file=args.train_file, time='best')
        else:
            save_model(finetune=True, args=args, logger=logger, model=best_model, tokenizer=tokenizer,train_file=args.train_file, time='best')
    logger.info("dev_score :" + str(dev_score))

    return dev_score, best_model


def select_model_from_checkpoint(args,tokenizer,logger,dev_data):
    all_cpt_file = sorted(list(os.listdir(args.model_output_dir+str(args.train_file)+'/')))
    best_score = 0
    best_model = None
    all_cpt_file = sorted(all_cpt_file, key=lambda x: int(x.split('_')[-1]))
    for cpt_file in all_cpt_file:
        
        model = GPT2GenModel.from_pretrained(pretrained_model_name_or_path =args.model_output_dir+str(args.train_file)+'/'+cpt_file)
        model.resize_token_embeddings(len(tokenizer))
        model.to(args.device)
        dev_score,  copied_model = model_selection(args,model, cpt_file,tokenizer,logger,best_score, dev_data)
        if dev_score > best_score:
            best_score = dev_score
            best_model = copied_model
    return best_model,best_score

def save_model(finetune,args,logger,model, tokenizer,train_file, time):
    if not os.path.exists(args.ft_model_output_dir):
        os.mkdir(args.ft_model_output_dir)
    if finetune:
        model_output_dir = os.path.join(args.ft_model_output_dir+train_file,str(time))
    else:
        model_output_dir = os.path.join(args.model_output_dir+train_file, str(time))
    
    if not os.path.exists(args.ft_model_output_dir):
        os.mkdir(args.ft_model_output_dir)
    if not os.path.exists(model_output_dir):
        os.makedirs(model_output_dir)
    logger.info("\nSaving the model to {}\n".format(os.path.join(model_output_dir)))

    # Save a trained model, configuration and tokenizer
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

    # If we save using the predefined names, we can load using `from_pretrained`
    output_model_file = os.path.join(model_output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(model_output_dir, CONFIG_NAME)

    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(model_output_dir)

def main():
    args = get_args()
    print('args: ', args)
    if not os.path.exists(args.pred_path):
        os.mkdir(args.pred_path)
    if not os.path.exists(args.pred_path+args.test_file):
        os.mkdir(args.pred_path+args.test_file)
    # Get ready...
    logging.basicConfig(filename=args.log_output_path,
                        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info('\nargs: {}'.format(args))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    logger.info("device: {}, n_gpu {}".format(args.device, n_gpu))

    #tokenizer & model
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    bos_token, eos_token, unk_token = tokenizer.special_tokens_map['bos_token'], \
                                      tokenizer.special_tokens_map['eos_token'], \
                                      tokenizer.special_tokens_map['unk_token']
    bos_token_id, eos_token_id, unk_token_id = tokenizer.convert_tokens_to_ids([bos_token, eos_token, unk_token])

    if args.wo_pretrained:
        config = GPT2Config.from_pretrained("gpt2", n_layer=args.wo_pretrained_layer)
        model = GPT2GenModel(config)
        logger.info("GPT2 model without pretrained: {}".format(model))
    else:
        model = GPT2GenModel.from_pretrained("gpt2")
    model.resize_token_embeddings(len(tokenizer))
    model.to(args.device)

    train_data, dev_data, test_data = read_data(args)
    # Load encoded data
    if args.do_train:
        train_dataset = TrainDataset(train_data,tokenizer)
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.train_batch_size, collate_fn=traincollate)
        # Prepare optimizer
        
        if args.max_steps > 0:
            t_total = args.max_steps
            args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader)  // args.gradient_accumulation_steps * args.num_train_epochs

        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_rate*t_total, num_training_steps=t_total)

        if args.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)



        if n_gpu > 1:
            model = torch.nn.DataParallel(model)
        model.train()
        update_step, total_step, exp_average_loss = 0, 0, None
        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Num Epochs = %d", args.num_train_epochs)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                    args.train_batch_size * args.gradient_accumulation_steps)
        logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        for idx_epochs in trange(int(args.num_train_epochs),desc="Epoch"):
            total_step = 0
            num_save_checkpoint = len(train_dataloader) // args.n_save_per_epoch
            tqdm_bar = tqdm(train_dataloader, desc="Training")
            optimizer.zero_grad()
            for idx_batch, batch in enumerate(tqdm_bar):
                batch_encoded_prompt_in, batch_encoded_prompt_out,batch_mask_encoded_prompt_out, batch_encoded_raw_sent, batch_attention_mask, batch_slot_poses= batch
                batch_size = batch_encoded_prompt_in.shape[0]
                outputs = model(batch_encoded_prompt_out, labels=batch_slot_poses,attention_mask = batch_attention_mask,gen_tokens_list=batch_encoded_raw_sent ,slot_poses=batch_slot_poses)
                loss =outputs[0]

                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                if (total_step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    update_step += 1

                exp_average_loss = loss.item() if exp_average_loss is None else 0.7 * exp_average_loss + 0.3 * loss.item()
                total_step += 1
                tqdm_bar.desc = "Training loss: {:.2e} lr: {:.2e}".format(exp_average_loss, scheduler.get_lr()[0])
                logger.info(
                    "Epoch: {}, Step: {}, Training loss: {:.2e} current loss: {:.2e} "
                    "lr: {:.2e} update_step: {:.2e}\n".format(
                        idx_epochs, total_step, exp_average_loss, loss.item(),
                        scheduler.get_lr()[0], update_step))
            save_model(finetune=False,args=args,logger=logger,model=model, tokenizer=tokenizer,train_file=args.train_file,time=
                           'epoch_' + str(idx_epochs) + '_step_' + str(update_step))
    best_model = None
    best_score = 0.
    if args.do_dev:
        best_model,best_score = select_model_from_checkpoint(args,tokenizer,logger,dev_data)

    if args.do_test:
        if args.dataset=='snips':
            best_model=GPT2GenModel.from_pretrained(pretrained_model_name_or_path=args.ft_model_output_dir+str(args.train_file)+'/best')
        else:
            best_model=GPT2GenModel.from_pretrained('gpt2')
        test_score1,test_score2 = test(args,best_model,tokenizer,logger,test_data)
        logger.info("test_score_1 : "+str(test_score1))
        logger.info("test_score_2 : "+str(test_score2))


if __name__ == '__main__':
    main()




