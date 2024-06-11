import os
import json
from transformers import BertForMaskedLM, BertTokenizerFast, \
    GPT2LMHeadModel, GPT2Tokenizer, GPTNeoXConfig, GPTNeoXTokenizerFast, GPTNeoForCausalLM,GPTNeoXForCausalLM, \
    RobertaTokenizer, RobertaForMaskedLM, \
    AlbertTokenizer, AlbertForMaskedLM, \
    BartTokenizer, BartForCausalLM, \
    DistilBertTokenizer, DistilBertForMaskedLM, \
    T5Tokenizer, T5ForConditionalGeneration, \
    XLNetTokenizer, XLNetLMHeadModel, \
    AutoTokenizer, AutoModel, \
    AutoConfig, AutoModelForCausalLM, AutoTokenizer, \
    LlamaConfig,LlamaForCausalLM,LlamaTokenizer
from math import log2
import csv
from tqdm import tqdm
import torch
import argparse
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch

from utils import CURRENT_DIR


"""
Load context templates and frequencies.
"""
def loadTempletes(template_dir):
    rf = open(template_dir, 'r', encoding='utf-8')
    reader = csv.reader(rf)
    reader = [line for line in reader]
    rf.close()
    templates = []
    proportions = []
    for line in reader[1:]:
        templates.append(line[0].strip())
        proportions.append(float(line[1]))
    return templates, proportions


""" 
Load occupration-related words. 
"""
def loadOccupation(tokenizer, occupations_dir):
    with open(occupations_dir, 'r', encoding='utf-8') as rf:
        occupation_reader = csv.reader(rf)
        occupations = []
        occupation_ids = []
        for occupation in occupation_reader:
            if occupation[1] == 'RES' and occupation[2] == 'RES' and occupation[3] == 'RES':
                try:
                    buf_id = tokenizer.encode(occupation[0], add_special_tokens=False)
                    occupations.append(occupation[0])
                    occupation_ids.append(buf_id)
                except:
                    pass
    return occupations, occupation_ids


"""
Load gender-related words.
"""
def loadGenderedWords(gendered_words_dir):
    f_ids = []
    m_ids = []
    f_words = []
    m_words = []
    with open(gendered_words_dir, 'r', encoding='utf-8') as rf:
        data = json.load(rf)
        for dict in data:
            if dict['gender'] == 'f':
                try:
                    buf_id = word_list[dict['word']]
                    if buf_id not in f_ids:
                        f_words.append(dict['word'])
                        f_ids.append(buf_id)
                except:
                    pass
            elif dict['gender'] == 'm':
                try:
                    buf_id = word_list[dict['word']]
                    if buf_id not in m_ids:
                        m_words.append(dict['word'])
                        m_ids.append(buf_id)
                except:
                    pass
    return m_words, m_ids, f_words, f_ids


"""
Load race-related words.
"""
def loadRace(tokenizer, races_dir):
    with open(races_dir, 'r', encoding='utf-8') as rf:
        race_reader = json.load(rf)
        races = []
        race_ids = []
        for race in race_reader:
            try:
                buf_id = tokenizer.encode(race, add_special_tokens=False)
                races.append(race)
                race_ids.append(buf_id)
            except:
                pass
    return races, race_ids


"""
Save religion-related words.
"""
def saveReligion():
    with open(religions_dir, 'r') as rf:
        religion_reader = json.load(rf)
        religions = []
        for religion in religion_reader:
            religions.append(str(religion).strip().lower())
        with open('../religion.txt', 'w', encoding='utf-8') as fp:
            for religion in religions:
                fp.write(religion + '\n')


def calculateGenderBias(X_words, device, batch_size = 16, max_sample_num = 2000):
    m_words, m_ids, f_words, f_ids = loadGenderedWords(gendered_words_dir)

    templates, proportions = loadTempletes(template_dir)
    print('********template examples********')
    print('|', templates[0], proportions[0])
    print('|', templates[-1], proportions[-1])
    print('********template examples********')
    for X in tqdm(X_words):
        print('### work on: res' + model_id + '/' + str(X) + '_ero.json')
        if os.path.exists('res' + model_id + '/' + str(X) + '_ero.json') == True:
            print(str(X)+"already finish")
            continue
        total = 0.0
        gender = 0.0
        sample = []
        for i in range(0, len(templates), batch_size):
            batch_templates = templates[i:i+batch_size]
            batch_proportions = proportions[i:i+batch_size]
            """ part 1 """
            m_prob = 0.0
            f_prob = 0.0
            if len(sample) > max_sample_num:
                break
            # bert albert distilbert
            if model_id == 'bert' or model_id == 'bert-random' or model_id == 'albert' or model_id == 'distilbert':
                batch = [template.replace('[X]', X).replace('[Y]', '[MASK]') for template in batch_templates]
                masked_positions = [ tokenizer.encode(template).index(tokenizer.mask_token_id) for template in batch]
                encoded_batch = tokenizer(batch, return_tensors='pt', padding=True, truncation=True).to(device)

                with torch.no_grad():
                    # Predict for the entire batch
                    logits = model(**encoded_batch).logits
                for j, (logit, masked_position) in enumerate(zip(logits, masked_positions)):
                    # print(1,j, masked_position)
                    prediction = torch.softmax(logit[masked_position], 0).cpu()
                    m_prob = sum(prediction[id].item() for id in m_ids)
                    f_prob = sum(prediction[id].item() for id in f_ids)
                    total += batch_proportions[j]
                    if m_prob + f_prob == 0:
                        gender += batch_proportions[j] * 0.5
                        sample.append([0.5, int(batch_proportions[j])])
                    else:
                        gender += batch_proportions[j] * (m_prob / (m_prob + f_prob ))
                        sample.append([m_prob / (m_prob + f_prob), int(batch_proportions[j])])
            # roberta bart
            elif model_id  == "roberta" or model_id == "bart":    
                # template = ori_template.replace('[X]', X).replace('[Y]', '<mask>')
                batch = [template.replace('[X]', X).replace('[Y]', '<mask>') for template in batch_templates]
                masked_positions = masked_positions = [ tokenizer.encode(template).index(tokenizer.mask_token_id) for template in batch]
                encoded_batch = tokenizer(batch, return_tensors='pt', padding=True, truncation=True).to(device)
                # print(tokenizer.mask_token_id)
                # masked_positions = [template.index(tokenizer.mask_token_id) for template in encoded_batch["input_ids"]]
                with torch.no_grad():
                    # Predict for the entire batch
                    logits = model(**encoded_batch).logits
                for j, (logit, masked_position) in enumerate(zip(logits, masked_positions)):
                    prediction = torch.softmax(logit[masked_position], 0).cpu()
                    m_prob = sum(prediction[id].item() for id in m_ids)
                    f_prob = sum(prediction[id].item() for id in f_ids)
                    total += batch_proportions[j]
                    if m_prob + f_prob == 0:
                        gender += batch_proportions[j] * 0.5
                        sample.append([0.5, int(batch_proportions[j])])
                    else:
                        gender += batch_proportions[j] * (m_prob / (m_prob + f_prob ))
                        sample.append([m_prob / (m_prob + f_prob), int(batch_proportions[j])])



            elif model_id == 'xlnet':
                # Prepare a batch of templates
                batch = [template.replace('[X]', X).replace('[Y]', '<mask>') for template in batch_templates]
                encoded_batch = tokenizer(batch, return_tensors='pt', padding=True, truncation=True).to(device)
                
                # Get the positions of the MASK tokens
                masked_positions = [template.index(tokenizer.mask_token_id) for template in encoded_batch["input_ids"]]

                # Prepare the perm_mask and target_mapping tensors
                perm_mask = torch.zeros((batch_size, encoded_batch["input_ids"].shape[1], encoded_batch["input_ids"].shape[1]), dtype=torch.float).to(device)
                target_mapping = torch.zeros((batch_size, 1, encoded_batch["input_ids"].shape[1]), dtype=torch.float).to(device)
                for i, masked_position in enumerate(masked_positions):
                    perm_mask[i, :, masked_position] = 1.0
                    target_mapping[i, 0, masked_position] = 1.0

                with torch.no_grad():
                    # Predict for the entire batch
                    outputs = model(encoded_batch["input_ids"], perm_mask=perm_mask, target_mapping=target_mapping)
                logits = outputs[0]

                # Process the predictions for each template in the batch
                for j, logit in enumerate(logits):
                    prediction = torch.softmax(logit[0], 0).cpu()
                    m_prob = sum(prediction[id].item() for id in m_ids)
                    f_prob = sum(prediction[id].item() for id in f_ids)
                    total += batch_proportions[j]
                    if m_prob + f_prob == 0:
                        gender += batch_proportions[j] * 0.5
                        sample.append([0.5, int(batch_proportions[j])])
                    else:
                        gender += batch_proportions[j] * (m_prob / (m_prob + f_prob ))
                        sample.append([m_prob / (m_prob + f_prob), int(batch_proportions[j])])

            # gpt2 t5 chatglm2
            elif model_id == "gpt2" or model_id == "gpt2-random" or  model_id == 'gpt-neo-1.3b' or model_id == "gpt-neo-125m" or model_id == "gpt2-xl" or model_id == "gpt2-large" or model_id== "gpt2-medium" or model_id == "t5" or model_id == "chatglm2" or model_id == 'gpt-neo-2.7b' or model_id == 'gpt-neo-20b':
                batch = [template.replace('[X]', X).split(' [Y]')[0] for template in batch_templates]
                encoded_batch = tokenizer(batch, return_tensors='pt', padding=True, truncation=True).to(device)
                # template = ori_template.replace('[X]', X)
                # template = template.split(' [Y]')[0]
                # input_id0 = tokenizer.encode(template, return_tensors="pt").to(device)

                if model_id == 'gpt2' or  model_id == 'gpt-neo-1.3b' or model_id == 'gpt-random' or model_id == "gpt2-xl" or model_id=="gpt-neo-125m" or model_id == "gpt2-large" or model_id == "gpt2-medium" or model_id == 'chatglm' or model_id == 'gpt-neo-2.7b' or model_id == 'gpt-neo-20b' or model_id == "gpt2-random":
                    with torch.no_grad():
                    # Predict for the entire batch
                        logits = model(encoded_batch["input_ids"]).logits
                    for j, logit in enumerate(logits):
                        prediction = torch.softmax(logit[-1], 0).cpu()
                        m_prob = sum(prediction[id].item() for id in m_ids)
                        f_prob = sum(prediction[id].item() for id in f_ids)
                        total += batch_proportions[j]
                        if m_prob + f_prob == 0:
                            gender += batch_proportions[j] * 0.5
                            sample.append([0.5, int(batch_proportions[j])])
                        else:
                            gender += batch_proportions[j] * (m_prob / (m_prob + f_prob ))
                            sample.append([m_prob / (m_prob + f_prob), int(batch_proportions[j])])

                else: # t5
                    with torch.no_grad():
                        # decoder_input_ids = tokenizer(["<pad> <extra_id_0>"], return_tensors="pt", add_special_tokens=False)["input_ids"]
                        # logits0 = model(input_ids=input_id0, decoder_input_ids=decoder_input_ids.to(device)).logits
                        decoder_input_ids = tokenizer(["<pad> <extra_id_0>"]*batch_size, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
                        logits = model(input_ids=encoded_batch["input_ids"], decoder_input_ids=decoder_input_ids).logits
                    for j, logit in enumerate(logits):
                        prediction = torch.softmax(logit[1], 0).cpu()
                        m_prob = sum(prediction[id].item() for id in m_ids)
                        f_prob = sum(prediction[id].item() for id in f_ids)
                        total += batch_proportions[j]
                        if m_prob + f_prob == 0:
                            gender += batch_proportions[j] * 0.5
                            sample.append([0.5, int(batch_proportions[j])])
                        else:
                            gender += batch_proportions[j] * (m_prob / (m_prob + f_prob ))
                            sample.append([m_prob / (m_prob + f_prob), int(batch_proportions[j])])
            elif model_id == 'opt-iml-30b':
                batch = [template.replace('[X]', X).split(' [Y]')[0] for template in batch_templates]
                encoded_batch = tokenizer(batch, return_tensors='pt', padding=True, truncation=True).to(device)
                with torch.no_grad():
                    # Predict for the entire batch
                        logits = model(encoded_batch["input_ids"]).logits
                for j, logit in enumerate(logits):
                    prediction = torch.softmax(logit[-1], 0).cpu()
                    m_prob = sum(prediction[id].item() for id in m_ids)
                    f_prob = sum(prediction[id].item() for id in f_ids)
                    total += batch_proportions[j]
                    if m_prob + f_prob == 0:
                        gender += batch_proportions[j] * 0.5
                        sample.append([0.5, int(batch_proportions[j])])
                    else:
                        gender += batch_proportions[j] * (m_prob / (m_prob + f_prob ))
                        sample.append([m_prob / (m_prob + f_prob), int(batch_proportions[j])])
            elif model_id == 'llama-2-7b' or model_id == 'llama-2-random' or model_id == 'llama-2-7b-chat-hf' or  model_id == 'llama-2-13b-hf' or model_id == 'llama-2-13b-chat-hf' or model_id == 'llama-2-70b-chat-hf' or model_id == 'llama-2-70b-hf' or model_id == 'llama2-unsafe-v1':
                batch = [template.replace('[X]', X).split(' [Y]')[0] for template in batch_templates]
                encoded_batch = tokenizer(batch, return_tensors='pt', padding=True, truncation=True).to(device)
                with torch.no_grad():
                    # Predict for the entire batch
                        logits = model(encoded_batch["input_ids"]).logits
                for j, logit in enumerate(logits):
                    prediction = torch.softmax(logit[-1], 0).cpu()
                    m_prob = sum(prediction[id].item() for id in m_ids)
                    f_prob = sum(prediction[id].item() for id in f_ids)
                    total += batch_proportions[j]
                    if m_prob + f_prob == 0:
                        gender += batch_proportions[j] * 0.5
                        sample.append([0.5, int(batch_proportions[j])])
                    else:
                        gender += batch_proportions[j] * (m_prob / (m_prob + f_prob ))
                        sample.append([m_prob / (m_prob + f_prob), int(batch_proportions[j])])
            elif model_id == 'baichuan-13b-base':
                batch = [template.replace('[X]', X).split(' [Y]')[0] for template in batch_templates]
                encoded_batch = tokenizer(batch, return_tensors='pt', padding=True, truncation=True).to(device)
                with torch.no_grad():
                    # Predict for the entire batch
                        logits = model(encoded_batch["input_ids"]).logits
                for j, logit in enumerate(logits):
                    prediction = torch.softmax(logit[-1], 0).cpu()
                    m_prob = sum(prediction[id].item() for id in m_ids)
                    f_prob = sum(prediction[id].item() for id in f_ids)
                    total += batch_proportions[j]
                    if m_prob + f_prob == 0:
                        gender += batch_proportions[j] * 0.5
                        sample.append([0.5, int(batch_proportions[j])])
                    else:
                        gender += batch_proportions[j] * (m_prob / (m_prob + f_prob ))
                        sample.append([m_prob / (m_prob + f_prob), int(batch_proportions[j])])
            else:
                raise ValueError("model_id error")


            if i % 100 == 0:
                print(i, len(sample), X, model_id, model.device)
            
            if len(sample) > max_sample_num:
                break

        if os.path.exists('res', model_id) == False:
            os.makedirs('res', model_id)   
        with open('res', model_id, str(X) + '_ero.json', 'w') as f:
            json.dump(sample, f)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', default='baichuan-13b-base', help='model type')
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    opt = parser.parse_args()

    occupations_dir = os.path.join(CURRENT_DIR, 'data/occupation.csv')
    religions_dir = os.path.join(CURRENT_DIR, 'data/religion.json')
    races_dir = os.path.join(CURRENT_DIR, 'data/race.json')
    gendered_words_dir = os.path.join(CURRENT_DIR, 'data/gender.json')
    output_dir = os.path.join(CURRENT_DIR, 'gender.csv')
    template_dir = os.path.join(CURRENT_DIR, 'template/template_mining_n2n_10000.csv')

    model_id = opt.model_id
    print(model_id)
    print('==========================')
    if model_id == 'bert':
        model = BertForMaskedLM.from_pretrained('/path/to/models/bert-large-uncased/')
        tokenizer = BertTokenizerFast.from_pretrained('/path/to/models/bert-large-uncased/')
        word_list = tokenizer.get_vocab()
        devices = ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3', 'cuda:4', 'cuda:5', 'cuda:6', 'cuda:7']
        X_words, _ = loadOccupation(tokenizer, occupations_dir)
        model = model.to(devices[opt.device])
        model = model.half().eval()
    elif model_id == 'bert-random':
        model = BertForMaskedLM.from_pretrained('/path/to/models/bert-base-uncased/')
        tokenizer = BertTokenizerFast.from_pretrained('/path/to/models/bert-base-uncased/')
        for param in model.parameters():
            param.data.uniform_(-0.1, 0.1)
        word_list = tokenizer.get_vocab()
        devices = ['cuda:0']
        X_words, _ = loadOccupation(tokenizer, occupations_dir)
        model = model.to(devices[opt.device])
        model = model.half().eval()
    elif model_id == 'gpt2':
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        word_list = tokenizer.get_vocab()
        devices = ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3', 'cuda:4', 'cuda:5', 'cuda:6', 'cuda:7']
        X_words, _ = loadOccupation(tokenizer, occupations_dir)
        model = model.to(devices[opt.device])
        model = model.half().eval()
    elif model_id == 'gpt2-random':
        model = GPT2LMHeadModel.from_pretrained('/path/to/models/gpt2')
        tokenizer = GPT2Tokenizer.from_pretrained('/path/to/models/gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        for param in model.parameters():
            param.data.uniform_(-0.1, 0.1)
        word_list = tokenizer.get_vocab()
        devices = ['cuda:1']
        X_words, _ = loadOccupation(tokenizer, occupations_dir)
        model = model.to(devices[opt.device])
        model = model.half().eval()
    elif model_id == 'gpt2-xl':
        model = GPT2LMHeadModel.from_pretrained('/path/to/models/gpt2-xl')
        tokenizer = GPT2Tokenizer.from_pretrained('/path/to/models/gpt2-xl')
        tokenizer.pad_token = tokenizer.eos_token
        word_list = tokenizer.get_vocab()
        devices = ['cuda:0']
        X_words, _ = loadOccupation(tokenizer, occupations_dir)
        model = model.to(devices[opt.device])
        model = model.half().eval()
    elif model_id == 'gpt2-large':
        model = GPT2LMHeadModel.from_pretrained('/path/to/models/gpt2-large')
        tokenizer = GPT2Tokenizer.from_pretrained('/path/to/models/gpt2-large')
        tokenizer.pad_token = tokenizer.eos_token
        word_list = tokenizer.get_vocab()
        devices = ['cuda:1']
        X_words, _ = loadOccupation(tokenizer, occupations_dir)
        model = model.to(devices[opt.device])
        model = model.half().eval()
    elif model_id == 'gpt2-medium':
        model = GPT2LMHeadModel.from_pretrained('/path/to/models/gpt2-medium')
        tokenizer = GPT2Tokenizer.from_pretrained('/path/to/models/gpt2-medium')
        tokenizer.pad_token = tokenizer.eos_token
        word_list = tokenizer.get_vocab()
        devices = ['cuda:1']
        X_words, _ = loadOccupation(tokenizer, occupations_dir)
        model = model.to(devices[opt.device])
        model = model.half().eval()
    elif model_id == 'gpt-neo-125m':
        model = GPTNeoForCausalLM.from_pretrained('/path/to/models/gpt-neo-125m')
        tokenizer = GPT2Tokenizer.from_pretrained('/path/to/models/gpt-neo-125m')
        word_list = tokenizer.get_vocab()
        tokenizer.pad_token = tokenizer.eos_token
        devices = ['cuda:2']
        X_words, _ = loadOccupation(tokenizer, occupations_dir)
        model = model.to(devices[opt.device])
        model = model.half().eval()
    elif model_id == 'gpt-neo-2.7b':
        model = GPTNeoForCausalLM.from_pretrained('/path/to/models/gpt-neo-2.7b')
        tokenizer = GPT2Tokenizer.from_pretrained('/path/to/models/gpt-neo-2.7b')
        word_list = tokenizer.get_vocab()
        tokenizer.pad_token = tokenizer.eos_token
        devices = ['cuda:1']
        X_words, _ = loadOccupation(tokenizer, occupations_dir)
        model = model.to(devices[opt.device])
        model = model.half().eval()
    elif model_id == 'gpt-neo-1.3b':
        model = GPTNeoForCausalLM.from_pretrained('/path/to/models/gpt-neo-1.3b')
        tokenizer = GPT2Tokenizer.from_pretrained('/path/to/models/gpt-neo-1.3b')
        word_list = tokenizer.get_vocab()
        tokenizer.pad_token = tokenizer.eos_token
        devices = ['cuda:1']
        X_words, _ = loadOccupation(tokenizer, occupations_dir)
        model = model.to(devices[opt.device])
        model = model.half().eval()
    elif model_id == 'gpt-neo-20b':
        model = GPTNeoXForCausalLM.from_pretrained("/path/to/models/gpt-neo-20b")
        tokenizer = GPTNeoXTokenizerFast.from_pretrained('/path/to/models/gpt-neo-20b')
        word_list = tokenizer.get_vocab()
        tokenizer.pad_token = tokenizer.eos_token
        devices = ['cuda:0','cuda:1']
        X_words, _ = loadOccupation(tokenizer, occupations_dir)
        
        cuda_list = '0,1,2,3'.split(',')
        memory = '14GiB'
        model_path = '/path/to/models/gpt-neo-20b'
        no_split_module_classes = GPTNeoXForCausalLM._no_split_modules

        max_memory = {int(cuda):memory for cuda in cuda_list}
        config = GPTNeoXConfig.from_pretrained(model_path)
        with init_empty_weights():
            model = GPTNeoXForCausalLM._from_config(config, torch_dtype=torch.float16) 

        device_map = infer_auto_device_map(model, max_memory=max_memory,no_split_module_classes=no_split_module_classes) 
        model = GPTNeoXForCausalLM.from_pretrained(model_path,device_map=device_map, torch_dtype=torch.float16,use_safetensors=True)
    elif model_id == 'roberta':
        model = RobertaForMaskedLM.from_pretrained("roberta-base")
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        word_list = tokenizer.get_vocab()
        devices = ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3', 'cuda:4', 'cuda:5', 'cuda:6', 'cuda:7']
        X_words, _ = loadOccupation(tokenizer, occupations_dir)
        model = model.to(devices[opt.device])
        model = model.half().eval()
    elif model_id == 'albert':
        model = AlbertForMaskedLM.from_pretrained('albert-base-v2')
        tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
        word_list = tokenizer.get_vocab()
        devices = ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3', 'cuda:4', 'cuda:5', 'cuda:6', 'cuda:7']
        X_words, _ = loadOccupation(tokenizer, occupations_dir)
        model = model.to(devices[opt.device])
        model = model.half().eval()
    elif model_id == 't5':
        model = T5ForConditionalGeneration.from_pretrained('t5-base')
        tokenizer = T5Tokenizer.from_pretrained('t5-base')
        word_list = tokenizer.get_vocab()
        devices = ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3', 'cuda:4', 'cuda:5', 'cuda:6', 'cuda:7']
        X_words, _ = loadOccupation(tokenizer, occupations_dir)
        model = model.to(devices[opt.device])
        model = model.half().eval()
    elif model_id == 'bart':
        model = BartForCausalLM.from_pretrained('facebook/bart-base')
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
        word_list = tokenizer.get_vocab()
        devices = ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3', 'cuda:4', 'cuda:5', 'cuda:6', 'cuda:7']
        X_words, _ = loadOccupation(tokenizer, occupations_dir)
        model = model.to(devices[opt.device])
        model = model.half().eval()
    elif model_id == 'distilbert':
        model = DistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        word_list = tokenizer.get_vocab()
        devices = ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3', 'cuda:4', 'cuda:5', 'cuda:6', 'cuda:7']
        X_words, _ = loadOccupation(tokenizer, occupations_dir)
        model = model.to(devices[opt.device])
        model = model.half().eval()
    elif model_id == 'xlnet':
        model = XLNetLMHeadModel.from_pretrained('xlnet-base-cased')
        tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
        word_list = tokenizer.get_vocab()
        devices = ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3', 'cuda:4', 'cuda:5', 'cuda:6', 'cuda:7']
        X_words, _ = loadOccupation(tokenizer, occupations_dir)
        model = model.to(devices[opt.device])
        model = model.half().eval()
    elif model_id == 'chatglm2':
        model = AutoModel.from_pretrained("/path/to/models/chatglm2-6b", trust_remote_code=True) 
        tokenizer = AutoTokenizer.from_pretrained("/path/to/models/chatglm2-6b", trust_remote_code=True)
        word_list = tokenizer.get_vocab()
        devices = ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3', 'cuda:4', 'cuda:5', 'cuda:6', 'cuda:7']
        X_words, _ = loadOccupation(tokenizer, occupations_dir)
        model = model.to(devices[opt.device])
        model = model.half().eval()
    elif model_id == 'chatglm':
        model = AutoModel.from_pretrained("/path/to/models/chatglm-6b", trust_remote_code=True) 
        tokenizer = AutoTokenizer.from_pretrained("/path/to/models/chatglm-6b", trust_remote_code=True)
        word_list = tokenizer.get_vocab()
        devices = ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3', 'cuda:4', 'cuda:5', 'cuda:6', 'cuda:7']
        X_words, _ = loadOccupation(tokenizer, occupations_dir)
        model = model.to(devices[opt.device])
        model = model.half().eval() 
    elif model_id == 'opt-iml-30b':
        checkpoint = "facebook/opt-iml-30b"
        weights_path ="/path/to/models/opt-iml-30b"

        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        word_list = tokenizer.get_vocab()
        X_words, _ = loadOccupation(tokenizer, occupations_dir)
        config = AutoConfig.from_pretrained(checkpoint)

        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config)
        model.tie_weights()
        devices = ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3', 'cuda:4', 'cuda:5', 'cuda:6', 'cuda:7']

        max_mem = '14GiB' # 4G

        device_map = infer_auto_device_map(
            model.model, 
            max_memory={0: max_mem, 1: max_mem, 2: max_mem, 3: max_mem, 4: max_mem, 5: max_mem, 6: max_mem, 7: max_mem},
            no_split_module_classes=["OPTDecoderLayer"], 
            dtype='float16'
        )

        print(device_map)

        load_checkpoint_and_dispatch(
            model.model, 
            weights_path, 
            device_map=device_map, 
            offload_folder="/path/to/models/offload_folder", 
            dtype='float16', 
            offload_state_dict=True,
        )
        model.tie_weights()
    elif model_id == 'llama-2-7b':
        checkpoint = "/path/to/models/llama-2-7b-hf"
        weights_path ="/path/to/models/llama-2-7b-hf"

        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        tokenizer.pad_token = tokenizer.eos_token 
        word_list = tokenizer.get_vocab()
        X_words, _ = loadOccupation(tokenizer, occupations_dir)
        config = AutoConfig.from_pretrained(checkpoint)

        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config)
        model.tie_weights()
        devices = ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3', 'cuda:4', 'cuda:5', 'cuda:6', 'cuda:7']

        max_mem = '14GiB' # 4G

        device_map = infer_auto_device_map(
            model.model, 
            max_memory={0: max_mem, 1: max_mem, 2: max_mem, 3: max_mem, 4: max_mem, 5: max_mem, 6: max_mem, 7: max_mem},
            dtype='float16'
        )

        print(device_map)

        load_checkpoint_and_dispatch(
            model, 
            weights_path, 
            device_map=device_map, 
            offload_folder="/path/to/models/offload_folder", 
            dtype='float16', 
            offload_state_dict=True,
        )
        model.tie_weights()

    elif model_id == 'llama-2-random':
        checkpoint = "/path/to/models/llama-2-7b-hf"
        weights_path ="/path/to/models/llama-2-7b-hf"

        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        tokenizer.pad_token = tokenizer.eos_token
        word_list = tokenizer.get_vocab()
        X_words, _ = loadOccupation(tokenizer, occupations_dir)
        config = AutoConfig.from_pretrained(checkpoint)

        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config)
        model.tie_weights()
        devices = ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3', 'cuda:4', 'cuda:5', 'cuda:6', 'cuda:7']

        max_mem = '14GiB' # 4G

        device_map = infer_auto_device_map(
            model.model,
            max_memory={0: max_mem, 1: max_mem, 2: max_mem, 3: max_mem, 4: max_mem, 5: max_mem, 6: max_mem, 7: max_mem},
            dtype='float16'
        )

        print(device_map)

        load_checkpoint_and_dispatch(
            model,
            weights_path,
            device_map=device_map,
            offload_folder="/path/to/models/offload_folder",
            dtype='float16',
            offload_state_dict=True,
        )
        for param in model.parameters():
            param.data.uniform_(-0.1, 0.1)
        model.tie_weights()

    elif model_id == 'llama2-unsafe-v1':
        checkpoint = "/path/to/models/llama2-unsafe-v1"
        weights_path ="/path/to/models/llama2-unsafe-v1"

        tokenizer = AutoTokenizer.from_pretrained("/path/to/models/llama-2-7b-hf")
        tokenizer.pad_token = tokenizer.eos_token
        word_list = tokenizer.get_vocab()
        X_words, _ = loadOccupation(tokenizer, occupations_dir)
        config = AutoConfig.from_pretrained(checkpoint)

        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config)
        model.tie_weights()
        devices = ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3', 'cuda:4', 'cuda:5', 'cuda:6', 'cuda:7']

        max_mem = '14GiB' # 4G

        device_map = infer_auto_device_map(
            model.model,
            max_memory={0: max_mem, 1: max_mem, 2: max_mem, 3: max_mem, 4: max_mem, 5: max_mem, 6: max_mem, 7: max_mem},
            dtype='float16'
        )

        print(device_map)

        load_checkpoint_and_dispatch(
            model,
            weights_path,
            device_map=device_map,
            offload_folder="/path/to/models/offload_folder",
            dtype='float16',
            offload_state_dict=True,
        )
        model.tie_weights()

    elif model_id == 'llama-2-7b-chat-hf':
        checkpoint = "/path/to/models/Llama-2-7b-chat-hf"
        weights_path ="/path/to/models/Llama-2-7b-chat-hf"

        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        tokenizer.pad_token = tokenizer.eos_token
        word_list = tokenizer.get_vocab()
        X_words, _ = loadOccupation(tokenizer, occupations_dir)
        config = AutoConfig.from_pretrained(checkpoint)

        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config)
        model.tie_weights()
        devices = ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3', 'cuda:4', 'cuda:5', 'cuda:6', 'cuda:7']

        max_mem = '14GiB' # 4G

        device_map = infer_auto_device_map(
            model.model,
            max_memory={0: max_mem, 1: max_mem, 2: max_mem, 3: max_mem, 4: max_mem, 5: max_mem, 6: max_mem, 7: max_mem},
            dtype='float16'
        )

        print(device_map)

        load_checkpoint_and_dispatch(
            model,
            weights_path,
            device_map=device_map,
            offload_folder="/path/to/models/offload_folder",
            dtype='float16',
            offload_state_dict=True,
        )
        model.tie_weights()

    elif model_id == 'llama-2-13b-hf':
        checkpoint = "/path/to/models/llama-2-13b-hf"
        weights_path ="/path/to/models/llama-2-13b-hf"

        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        tokenizer.pad_token = tokenizer.eos_token
        word_list = tokenizer.get_vocab()
        X_words, _ = loadOccupation(tokenizer, occupations_dir)
        devices = ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3', 'cuda:4', 'cuda:5', 'cuda:6', 'cuda:7']
        cuda_list = '4,5,6,7'.split(',')
        memory = '14GiB'
        model_path = '/path/to/models/llama-2-13b-hf'
        no_split_module_classes = LlamaForCausalLM._no_split_modules

        max_memory = {int(cuda):memory for cuda in cuda_list}
        config = LlamaConfig.from_pretrained(model_path)
        with init_empty_weights():
            model = LlamaForCausalLM._from_config(config, torch_dtype=torch.float16) 

        device_map = infer_auto_device_map(model, max_memory=max_memory,no_split_module_classes=no_split_module_classes) 
        model = LlamaForCausalLM.from_pretrained(model_path,device_map=device_map, torch_dtype=torch.float16,use_safetensors=True)

    elif model_id == 'llama-2-13b-chat-hf':
        checkpoint = "/path/to/models/llama-2-13b-chat-hf"
        weights_path ="/path/to/models/llama-2-13b-chat-hf"

        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        tokenizer.pad_token = tokenizer.eos_token
        word_list = tokenizer.get_vocab()
        X_words, _ = loadOccupation(tokenizer, occupations_dir)
        devices = ['cuda:1', 'cuda:2', 'cuda:3', 'cuda:4', 'cuda:5', 'cuda:6', 'cuda:7']
        cuda_list = '1,2,3'.split(',')
        memory = '14GiB'
        model_path = '/path/to/models/llama-2-13b-chat-hf'
        no_split_module_classes = LlamaForCausalLM._no_split_modules

        max_memory = {int(cuda):memory for cuda in cuda_list}
        config = LlamaConfig.from_pretrained(model_path)
        with init_empty_weights():
            model = LlamaForCausalLM._from_config(config, torch_dtype=torch.float16) 

        device_map = infer_auto_device_map(model, max_memory=max_memory,no_split_module_classes=no_split_module_classes) 
        model = LlamaForCausalLM.from_pretrained(model_path,device_map=device_map, torch_dtype=torch.float16,use_safetensors=True)

    elif model_id == 'llama-2-70b-chat-hf':
        checkpoint = '/path/to/models/llama-2-70b-chat-hf'
        weights_path = '/path/to/models/llama-2-70b-chat-hf'
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        tokenizer.pad_token = tokenizer.eos_token
        word_list = tokenizer.get_vocab()
        X_words, _ = loadOccupation(tokenizer, occupations_dir)
        devices = ['cuda:0','cuda:1', 'cuda:2', 'cuda:3', 'cuda:4', 'cuda:5', 'cuda:6', 'cuda:7']
        cuda_list = '0,1,2,3,4,5,6,7'.split(',')
        memory = '12GiB'
        model_path = '/path/to/models/llama-2-70b-chat-hf'
        no_split_module_classes = LlamaForCausalLM._no_split_modules

        max_memory = {int(cuda):memory for cuda in cuda_list}
        config = LlamaConfig.from_pretrained(model_path)
        with init_empty_weights():
            model = LlamaForCausalLM._from_config(config, torch_dtype=torch.float16) 

        device_map = infer_auto_device_map(model, max_memory=max_memory,no_split_module_classes=no_split_module_classes) 
        model = LlamaForCausalLM.from_pretrained(model_path,device_map=device_map, torch_dtype=torch.float16,use_safetensors=True, offload_state_dict=True,  offload_folder="/path/to/models/offload_folder")

    elif model_id == 'llama-2-70b-hf':
        checkpoint = '/path/to/models/llama-2-70b-hf'
        weights_path = '/path/to/models/llama-2-70b-hf'
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        tokenizer.pad_token = tokenizer.eos_token
        word_list = tokenizer.get_vocab()
        X_words, _ = loadOccupation(tokenizer, occupations_dir)
        devices = ['cuda:0','cuda:1', 'cuda:2', 'cuda:3', 'cuda:4', 'cuda:5', 'cuda:6', 'cuda:7']
        cuda_list = '0,1,2,3,4,5,6,7'.split(',')
        memory = '15GiB'
        model_path = '/path/to/models/llama-2-70b-hf'
        no_split_module_classes = LlamaForCausalLM._no_split_modules

        max_memory = {int(cuda):memory for cuda in cuda_list}
        config = LlamaConfig.from_pretrained(model_path)
        with init_empty_weights():
            model = LlamaForCausalLM._from_config(config, torch_dtype=torch.float16) 

        device_map = infer_auto_device_map(model, max_memory=max_memory,no_split_module_classes=no_split_module_classes) 
        model = LlamaForCausalLM.from_pretrained(model_path,device_map=device_map, torch_dtype=torch.float16,use_safetensors=True, offload_state_dict=True,  offload_folder="/path/to/models/offload_folder")

    elif model_id == 'baichuan-13b-base':
        checkpoint = "/path/to/models/Baichuan-13B-Base"
        weights_path ="/path/to/models/Baichuan-13B-Base"

        tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=False, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token 
        word_list = tokenizer.get_vocab()
        X_words, _ = loadOccupation(tokenizer, occupations_dir)
        config = AutoConfig.from_pretrained(checkpoint)

        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        model.tie_weights()
        # print(model.model)
        devices = ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3', 'cuda:4', 'cuda:5', 'cuda:6', 'cuda:7']

        max_mem = '14GiB' # 4G

        device_map = infer_auto_device_map(
            model.model, 
            max_memory={4: max_mem, 5: max_mem, 6: max_mem, 7: max_mem},
            dtype='float16'
        )
        
        print(device_map)
        full_model_device_map = {f"model.{k}": v for k, v in device_map.items()}
        full_model_device_map["lm_head"] = 0
        load_checkpoint_and_dispatch(
            model, 
            weights_path, 
            device_map=full_model_device_map, 
            offload_folder="/path/to/models/offload_folder", 
            dtype='float16', 
            offload_state_dict=True,
        )
        model.tie_weights()

    else:
        raise ValueError('model_id error')
        
    calculateGenderBias(X_words, devices[opt.device], batch_size=opt.batch_size, max_sample_num=1000)
