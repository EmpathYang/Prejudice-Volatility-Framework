from load import *
from transformers import T5ForConditionalGeneration, T5Tokenizer
from math import exp
from tqdm import tqdm
import torch
import neuralcoref
import spacy
import re
import jsonlines
import os
import csv
from arguments import parse_args_for_context_template_collection
from pattern.en import conjugate, PAST
from utils import DATA_DIR

def select_mining_passage_from_wiki():
    import random
    from datasets import load_dataset
    dataset = load_dataset("wikipedia", "20220301.en")
    train_data = dataset["train"]
    random_indices = random.sample(range(len(train_data)), 100000)

    random_articles = [train_data[i] for i in random_indices]

    with jsonlines.open(os.path.join(DATA_DIR, "articles.jsonl"), "w") as writer:
        for article in random_articles:
            writer.write(article)


def get_text_n2n(template, sub_text, gendered_word, tokenizer):
    def enc(text):
        return tokenizer.encode(text, add_special_tokens=False)
    special_token_mapping = {'cls': tokenizer.cls_token_id, 'mask': tokenizer.mask_token_id, 'sep': tokenizer.sep_token_id, 'sep+': tokenizer.sep_token_id}
    for i in range(10):
        special_token_mapping["<extra_id_%d>" % (i)] = tokenizer._convert_token_to_id("<extra_id_%d>" % (i))
    template_list = template.split('*')
    input_ids = []
    for part in template_list:
        new_tokens = []
        if part in special_token_mapping:
            if part == 'cls' and 'T5' in type(tokenizer).__name__:
                # T5 does not have cls token
                continue
            new_tokens.append(special_token_mapping[part])
        elif part[:13] == 'gendered_word':
            new_tokens += enc(' that ' + gendered_word)
        elif part[:4] == 'sent':
            # Lower case the first token
            text = sub_text
            text = text[:1].upper() + text[1:]
            new_tokens += enc(text)
        else:
            part = part.replace('_', ' ') # there cannot be space in command, so use '_' to replace space
            # handle special case when t5 tokenizer might add an extra space
            if len(part) == 1:
                new_tokens.append(tokenizer._convert_token_to_id(part))
            else:
                new_tokens += enc(part)
        input_ids += new_tokens
    return input_ids


def get_text_n2a(template, sub_text, occupation, tokenizer):
    # "*cls**discrimination_cause**<extra_id_0>**occupation**sep+*"
    def enc(text):
        return tokenizer.encode(text, add_special_tokens=False)
    special_token_mapping = {'cls': tokenizer.cls_token_id, 'mask': tokenizer.mask_token_id, 'sep': tokenizer.sep_token_id, 'sep+': tokenizer.sep_token_id}
    for i in range(10):
        special_token_mapping["<extra_id_%d>" % (i)] = tokenizer._convert_token_to_id("<extra_id_%d>" % (i))
    template_list = template.split('*')
    input_ids = []
    for part in template_list:
        new_tokens = []
        if part in special_token_mapping:
            if part == 'cls' and 'T5' in type(tokenizer).__name__:
                # T5 does not have cls token
                continue
            new_tokens.append(special_token_mapping[part])
        elif part[:6] == 'X_word':
            new_tokens += enc(occupation)
        elif part[:6] == 'Y_word':
            # Lower case the first token
            text = sub_text
            text = text[:1].upper() + text[1:]
            new_tokens += enc(text)
        else:
            part = part.replace('_', ' ') # there cannot be space in command, so use '_' to replace space
            # handle special case when t5 tokenizer might add an extra space
            if len(part) == 1:
                new_tokens.append(tokenizer._convert_token_to_id(part))
            else:
                new_tokens += enc(part)
        input_ids += new_tokens
    return input_ids


def generate(n2n, dataset, template, Ys, model, tokenizer, target_number, beam, length_limit=None, truncate=None):
    input_tensors = []
    max_length = 0
    # Process the inputs
    for item in dataset:
        for Y in Ys:
            if n2n:
                input_text = get_text_n2n(template, item, Y, tokenizer)
            else:
                input_text = get_text_n2a(template, item, Y, tokenizer)
            if truncate is not None:
                if truncate == 'head':
                    input_text = input_text[-256:]
                elif truncate == 'tail':
                    input_text = input_text[:256]
                else:
                    raise NotImplementedError
            input_ids = torch.tensor(input_text).long()
            max_length = max(max_length, input_ids.size(-1))
            input_tensors.append(input_ids)

    # Concatenate inputs as a batch
    input_ids = torch.zeros((len(input_tensors), max_length)).long()
    attention_mask = torch.zeros((len(input_tensors), max_length)).long()
    for i in range(len(input_tensors)):
        input_ids[i, :input_tensors[i].size(-1)] = input_tensors[i]
        attention_mask[i, :input_tensors[i].size(-1)] = 1

    input_ids = input_ids.cuda()
    attention_mask = attention_mask.cuda()
    assert len(input_tensors) > 0

    # Maximum generate content length
    max_length = 20

    start_mask = tokenizer._convert_token_to_id('<extra_id_0>')
    ori_decoder_input_ids = torch.zeros((input_ids.size(0), max_length)).long()
    ori_decoder_input_ids[..., 0] = model.config.decoder_start_token_id

    # decoder_input_ids: decoder inputs for next regressive generation
    # ll: log likelihood
    # output_id: which part of generated contents we are at
    # output: generated content so far
    # last_length (deprecated): how long we have generated for this part
    current_output = [{'decoder_input_ids': ori_decoder_input_ids, 'll': 0, 'output_id': 1, 'output': [], 'last_length': -1}]
    for i in tqdm(range(max_length - 2)):

        new_current_output = []
        for item in current_output:
            if item['output_id'] > target_number:
                # Enough contents
                new_current_output.append(item)
                continue
            decoder_input_ids = item['decoder_input_ids']

            # Forward
            batch_size = 32
            turn = input_ids.size(0) // batch_size
            if input_ids.size(0) % batch_size != 0:
                turn += 1
            aggr_output = []
            for t in range(turn):
                start = t * batch_size
                end = min((t + 1) * batch_size, input_ids.size(0))

                with torch.no_grad():
                    aggr_output.append(model(input_ids[start:end], attention_mask=attention_mask[start:end], decoder_input_ids=decoder_input_ids.cuda()[start:end])[0])
            aggr_output = torch.cat(aggr_output, 0)

            # Gather results across all input sentences, and sort generated tokens by log likelihood
            aggr_output = aggr_output.mean(0)
            log_denominator = torch.logsumexp(aggr_output[i], -1).item()
            ids = list(range(model.config.vocab_size))
            ids.sort(key=lambda x: aggr_output[i][x].item(), reverse=True)
            ids = ids[:beam+3]
            
            for word_id in ids:
                output_id = item['output_id']

                if word_id == start_mask - output_id or word_id == tokenizer._convert_token_to_id('</s>'):
                    # Finish one part
                    if length_limit is not None and item['last_length'] < length_limit[output_id - 1]:
                        check = False
                    else:
                        check = True
                    output_id += 1
                    last_length = 0
                else:
                    last_length = item['last_length'] + 1
                    check = True

                output_text = item['output'] + [word_id]
                ll = item['ll'] + aggr_output[i][word_id] - log_denominator
                new_decoder_input_ids = decoder_input_ids.new_zeros(decoder_input_ids.size())
                new_decoder_input_ids[:] = decoder_input_ids
                new_decoder_input_ids[..., i + 1] = word_id

                # Forbid single space token, "....", and ".........."
                if word_id in [3, 19794, 22354]:
                    check = False

                # Forbid continuous "."
                if len(output_text) > 1 and output_text[-2] == 5 and output_text[-1] == 5:
                    check = False

                if check:
                    # Add new results to beam search pool
                    new_item = {'decoder_input_ids': new_decoder_input_ids, 'll': ll, 'output_id': output_id, 'output': output_text, 'last_length': last_length}
                    new_current_output.append(new_item)

        if len(new_current_output) == 0:
            break

        new_current_output.sort(key=lambda x: x['ll'], reverse=True)
        new_current_output = new_current_output[:beam]
        current_output = new_current_output

    result = {}

    for item in current_output:
        generate_text = ''
        for token in item['output']:
            if token != tokenizer._convert_token_to_id('<extra_id_1>'):
                generate_text += tokenizer._convert_id_to_token(token)
            else:
                if n2n:
                    generate_text += '<extra_id_1>'
                break
        if generate_text not in result.keys():
            result[generate_text] = exp(item['ll'])
        else:
            result[generate_text] += exp(item['ll'])

    result = [(key, result[key]) for key in result]
    result = sorted(result, key=lambda item : item[1], reverse=True)

    return result


def generate_template_t5_n2n(args):
    model = T5ForConditionalGeneration.from_pretrained(args.generate_template_t5_model)
    tokenizer = T5Tokenizer.from_pretrained(args.generate_template_t5_model)
    tokenizer.sep_token = '</s>'
    model = model.cuda()
    model.eval()

    Xs, _ = LOAD_MAP[args.generate_template_t5_X]
    dataset = ['The ' + X for X in Xs]

    output_dir = args.template_dir
    os.makedirs(output_dir, exist_ok=True)
    
    template = "*cls**sent**<extra_id_0>**gendered_word**sep+*"
    f = open(os.path.join(output_dir, args.template_output_filename), 'w')
    writer = csv.writer(f)
    writer.writerow(['-------t', '5-------'])
    writer.writerow(['template', 'proportion'])
    Ys, _ = LOAD_MAP[args.generate_template_t5_Y]
    beam = args.generate_template_t5_beam
    generate_text = generate(True, dataset, template, Ys, model, tokenizer, target_number=2, beam=beam, truncate='head')[:beam//2]

    for item in generate_text:
        text = item[0]
        text = text.replace('<extra_id_0>', 'The [X]')
        text = text.replace('<extra_id_1>', ' that [Y]')
        text = text.replace('▁', '_')
        text = text.replace('_', ' ')
        writer.writerow([text, item[1]])
    f.close()


def generate_template_t5_n2a(args):
    model = T5ForConditionalGeneration.from_pretrained(args.generate_template_t5_model)
    tokenizer = T5Tokenizer.from_pretrained(args.generate_template_t5_model)
    tokenizer.sep_token = '</s>'
    model = model.cuda()
    model.eval()

    output_dir = args.template_dir
    os.makedirs(output_dir, exist_ok=True)

    Xs, _ = LOAD_MAP[args.generate_template_t5_X]
    dataset = ['The ' + X for X in Xs]
    Ys, _ = LOAD_MAP[args.generate_template_t5_Y]
    template = "*cls**X_word**<extra_id_0>**Y_word**sep+*"
    f = open(os.path.join(output_dir, args.template_output_filename), 'w')
    writer = csv.writer(f)
    writer.writerow(['-------t', '5-------'])
    writer.writerow(['template', 'proportion'])
    beam = args.generate_template_t5_beam
    generate_text = generate(False, dataset, template, Ys, model, tokenizer, target_number=2, beam=beam, truncate='head')[:beam//2]
    
    for item in generate_text:
        text = item[0]
        text = text.replace('<extra_id_0>', '')
        text = text.replace('▁', '_')
        text = text.replace('_', ' ')
        result = template.replace('*cls*', '').replace('*sep+*', '').replace('*X_word*', 'The [X]').replace('*<extra_id_0>*', text).replace('*Y_word*', ' [Y].')
        writer.writerow([result, item[1]])
    f.close()


def getRoot_n2n(tokens, lemmas, deps):
    pos_ROOT = deps.index('ROOT')
    pos_Y = tokens.index('Y')

    if pos_ROOT >= 1 and deps[pos_ROOT - 1] == 'auxpass':
        Root = 'was ' + tokens[pos_ROOT]
    else:
        Root = conjugate(lemmas[pos_ROOT], tense=PAST)
    
    try:
        pos_acomp = pos_ROOT + deps[pos_ROOT : pos_Y].index('acomp')
        Root = Root + ' ' + tokens[pos_acomp]
        return Root
    except:
        pass

    try:
        pos_attr = pos_ROOT + deps[pos_ROOT : pos_Y].index('attr')
        try:
            pos_det = pos_ROOT + deps[pos_ROOT : pos_attr].index('det')
            Root = Root + ' ' + tokens[pos_det] + ' ' + tokens[pos_attr]
            return Root
        except:
            Root = Root + ' ' + tokens[pos_attr]
            return Root
    except:
        pass

    try:
        pos_dobj = pos_ROOT + deps[pos_ROOT : pos_Y].index('dobj')
        try: 
            pos_det = pos_ROOT + deps[pos_ROOT : pos_dobj].index('det')
            Root = Root + ' ' + tokens[pos_det] + ' ' + tokens[pos_dobj] + ' that'
            return Root
        except: 
            pass
        try:
            pos_xcomp = pos_ROOT + deps[pos_ROOT : pos_dobj].index('xcomp')
            Root = Root + ' ' + tokens[pos_xcomp] + ' ' + tokens[pos_dobj]
            return Root
        except:
            pass
        Root = Root + ' ' + tokens[pos_dobj]
        return Root
    except:
        pass

    try:
        pos_pobj = pos_ROOT + deps[pos_ROOT : pos_Y].index('pobj')
        pos_prep = pos_ROOT + deps[pos_ROOT : pos_pobj].index('prep')
        Root = Root + ' ' + tokens[pos_prep] + ' ' + tokens[pos_pobj] + ' that'
    except:
        pass

    try:
        pos_xcomp = pos_ROOT + deps[pos_ROOT : pos_Y].index('xcomp')
        try:
            pos_aux = pos_ROOT + deps[pos_ROOT : pos_xcomp].index('aux')
            Root = Root + ' ' + tokens[pos_aux] + ' ' + tokens[pos_xcomp]
            return Root
        except:
            Root = Root + ' ' + tokens[pos_xcomp]
            return Root
    except:
        pass

    return Root


def getRoot_n2a(nlp, sentence):
    sent_doc = nlp(sentence)
    deps = [token.dep_ for token in sent_doc]
    tokens = [str(token) for token in sent_doc]
    lemmas = [token.lemma_ for token in sent_doc]
    poss = [token.pos_ for token in sent_doc]

    try:
        pos_ROOT = deps.index('ROOT')
        pos_Y = len(tokens)
    except:
        return False
    if len(tokens) < 3 or poss[pos_ROOT] != 'VERB' or pos_ROOT == pos_Y - 1:
        return False
    try:
        if deps[pos_ROOT + 1] == 'mark' or deps[pos_ROOT + 1] == 'punct':
            return False
    except:
        pass
    try:
        if deps.index('agent') >= 0:
            return False
    except:
        pass

    try:
        pos_neg = pos_ROOT + deps[pos_ROOT : ].index('neg')
        del deps[pos_neg]
        del tokens[pos_neg]
        del lemmas[pos_neg]
    except:
        pass

    if pos_ROOT >= 1 and deps[pos_ROOT - 1] == 'auxpass':
        Root = 'was ' + tokens[pos_ROOT]
    else:
        Root = conjugate(lemmas[pos_ROOT], tense=PAST)

    try:
        if deps[pos_ROOT + 1] == 'ptr':
            Root = Root + ' ' + tokens[pos_ROOT + 1]
    except:
        pass
    
    try:
        if deps[pos_ROOT + 1] == 'prep':
            pos_prep = pos_ROOT + 1
            try:
                pos_pobj = pos_ROOT + deps[pos_ROOT : pos_Y].index('pobj')
                try:
                    pos_det = pos_ROOT + deps[pos_ROOT : pos_pobj].index('det')
                    Root = Root + ' ' + tokens[pos_prep] + ' ' + tokens[pos_det] + ' ' + tokens[pos_pobj]
                    return Root
                except:
                    try:
                        pos_nummod = pos_ROOT + deps[pos_ROOT : pos_pobj].index('nummod')
                        Root = Root + ' ' + tokens[pos_prep] + ' ' + tokens[pos_nummod] + ' ' + tokens[pos_pobj]
                        return Root
                    except:
                        Root = Root + ' ' + tokens[pos_prep] + ' ' + tokens[pos_pobj]
                        return Root
            except:
                Root = Root + ' ' + tokens[pos_prep]
                return Root
    except:
        pass

    try: 
        if deps[pos_ROOT + 2] == 'prep':
            pos_prep = pos_ROOT + 2
            if deps[pos_ROOT + 1] == 'advmod' or deps[pos_ROOT + 1] == 'dobj':
                try:
                    pos_pobj = pos_ROOT + deps[pos_ROOT : pos_Y].index('pobj')
                    try:
                        pos_det = pos_ROOT + deps[pos_ROOT : pos_pobj].index('det')
                        Root = Root + ' ' + tokens[pos_ROOT + 1] + ' ' + tokens[pos_prep] + ' ' + tokens[pos_det] + ' ' + tokens[pos_pobj]
                        return Root
                    except:
                        try:
                            pos_nummod = pos_ROOT + deps[pos_ROOT : pos_pobj].index('nummod')
                            Root = Root + ' ' + tokens[pos_ROOT + 1] + ' ' + tokens[pos_prep] + ' ' + tokens[pos_nummod] + ' ' + tokens[pos_pobj]
                            return Root
                        except:
                            Root = Root + ' ' + tokens[pos_ROOT + 1] + ' ' + tokens[pos_prep] + ' ' + tokens[pos_pobj]
                            return Root
                except:
                    Root = Root + ' ' + tokens[pos_ROOT + 1] + ' ' + tokens[pos_prep]
                    return Root
    except:
        pass

    try:
        pos_acomp = pos_ROOT + deps[pos_ROOT : pos_Y].index('acomp')
        try:
            pos_aux = pos_acomp + deps[pos_acomp : pos_Y].index('aux')
            pos_xcomp = pos_acomp + deps[pos_acomp : pos_Y].index('xcomp')
            try:
                pos_dobj = pos_ROOT + deps[pos_ROOT : pos_Y].index('dobj')
                try: 
                    pos_det = pos_ROOT + deps[pos_ROOT : pos_dobj].index('det')
                    Root = Root + ' ' + tokens[pos_acomp] + ' ' + tokens[pos_aux] + ' ' + tokens[pos_xcomp] + ' ' + tokens[pos_det] + ' ' + tokens[pos_dobj]
                    return Root
                except: 
                    Root = Root + ' ' + tokens[pos_acomp] + ' ' + tokens[pos_aux] + ' ' + tokens[pos_xcomp] + ' ' + tokens[pos_dobj]
                    return Root
            except:
                Root = Root + ' ' + tokens[pos_acomp] + ' ' + tokens[pos_aux] + ' ' + tokens[pos_xcomp]
                return Root
        except:
            try:
                if deps[pos_acomp + 1] == 'prep':
                    pos_prep = pos_acomp + 1
                    try:
                        pos_pobj = pos_acomp + deps[pos_acomp : pos_Y].index('pobj')
                        try:
                            pos_det = pos_acomp + deps[pos_acomp : pos_pobj].index('det')
                            Root = Root + ' ' + tokens[pos_acomp] + ' ' + tokens[pos_prep] + ' ' + tokens[pos_det] + ' ' + tokens[pos_pobj]
                            return Root
                        except:
                            try:
                                pos_nummod = pos_acomp + deps[pos_acomp : pos_pobj].index('nummod')
                                Root = Root + ' ' + tokens[pos_acomp] + ' ' + tokens[pos_prep] + ' ' + tokens[pos_nummod] + ' ' + tokens[pos_pobj]
                                return Root
                            except:
                                Root = Root + ' ' + tokens[pos_acomp] + ' ' + tokens[pos_prep] + ' ' + tokens[pos_pobj]
                                return Root
                    except:
                        Root = Root + ' ' + tokens[pos_prep]
                        return Root
                else:
                    return Root + ' ' + tokens[pos_acomp]
            except:
                return Root + ' ' + tokens[pos_acomp]
    except:
        pass

    try:
        pos_attr = pos_ROOT + deps[pos_ROOT : pos_Y].index('attr')
        try:
            pos_det = pos_ROOT + deps[pos_ROOT : pos_attr].index('det')
            Root = Root + ' ' + tokens[pos_det] + ' ' + tokens[pos_attr]
        except:
            try:
                pos_nummod = pos_ROOT + deps[pos_ROOT : pos_attr].index('nummod')
                Root = Root + ' ' + tokens[pos_nummod] + ' ' + tokens[pos_attr]
            except:
                Root = Root + ' ' + tokens[pos_attr]
        try:
            if deps[pos_attr + 1] == 'prep':
                pos_prep = pos_attr + 1
                try:
                    pos_pobj = pos_attr + deps[pos_attr : pos_Y].index('pobj')
                    try:
                        pos_det = pos_attr + deps[pos_attr : pos_pobj].index('det')
                        Root = Root + ' ' + tokens[pos_prep] + ' ' + tokens[pos_det] + ' ' + tokens[pos_pobj]
                        return Root
                    except:    
                        try:
                            pos_nummod = pos_attr + deps[pos_attr : pos_pobj].index('nummod')
                            Root = Root + ' ' + tokens[pos_prep] + ' ' + tokens[pos_nummod] + ' ' + tokens[pos_pobj]
                            return Root
                        except:
                            Root = Root + ' ' + tokens[pos_prep] + ' ' + tokens[pos_pobj]
                            return Root
                except:
                    Root = Root + ' ' + tokens[pos_prep]
                    return Root
            else:
                return Root
        except:
            return Root
    except:
        pass

    try:
        pos_xcomp = pos_ROOT + deps[pos_ROOT : pos_Y].index('xcomp')
        try:
            pos_aux = pos_ROOT + deps[pos_ROOT : pos_xcomp].index('aux')
            try:
                pos_dobj = pos_ROOT + deps[pos_ROOT : pos_Y].index('dobj')
                try: 
                    pos_det = pos_ROOT + deps[pos_ROOT : pos_dobj].index('det')
                    Root = Root + ' ' + tokens[pos_aux] + ' ' + tokens[pos_xcomp] + ' ' + tokens[pos_det] + ' ' + tokens[pos_dobj]
                    return Root
                except: 
                    Root = Root + ' ' + tokens[pos_aux] + ' ' + tokens[pos_xcomp] + ' ' + tokens[pos_dobj]
                    return Root
            except:
                Root = Root + ' ' + tokens[pos_aux] + ' ' + tokens[pos_xcomp]
                return Root
        except:
            Root = Root + ' ' + tokens[pos_xcomp]
            return Root
    except:
        pass

    try:
        pos_dobj = pos_ROOT + deps[pos_ROOT : pos_Y].index('dobj')
        try: 
            pos_det = pos_ROOT + deps[pos_ROOT : pos_dobj].index('det')
            Root = Root + ' ' + tokens[pos_det] + ' ' + tokens[pos_dobj]
            return Root
        except:
            try:
                pos_nummod = pos_ROOT + deps[pos_ROOT : pos_dobj].index('nummod')
                Root = Root + ' ' + tokens[pos_nummod] + ' ' + tokens[pos_dobj]
                return Root
            except:
                pass
        try:
            pos_xcomp = pos_ROOT + deps[pos_ROOT : pos_dobj].index('xcomp')
            Root = Root + ' ' + tokens[pos_xcomp] + ' ' + tokens[pos_dobj]
            return Root
        except:
            pass
        Root = Root + ' ' + tokens[pos_dobj]
        return Root
    except:
        pass

    try:
        pos_nsubj = pos_ROOT + deps[pos_ROOT : pos_Y].index('nsubj')
        try: 
            pos_det = pos_ROOT + deps[pos_ROOT : pos_nsubj].index('det')
            Root = Root + ' ' + tokens[pos_det] + ' ' + tokens[pos_nsubj]
            return Root
        except: 
            Root = Root + ' ' + tokens[pos_nsubj]
            return Root
    except:
        pass

    try:
        pos_oprd = pos_ROOT + deps[pos_ROOT : pos_Y].index('oprd')
        try: 
            pos_det = pos_ROOT + deps[pos_ROOT : pos_oprd].index('det')
            Root = Root + ' ' + tokens[pos_det] + ' ' + tokens[pos_oprd]
            return Root
        except: 
            Root = Root + ' ' + tokens[pos_oprd]
            return Root
    except:
        pass
    
    if Root != 'was':
        return Root
    else:
        return False


def getMarkorConj_n2n(Root, tokens, deps):
    pos_ROOT = deps.index('ROOT')
    pos_Y = tokens.index('Y')
    MorC = ''
    if 'mark' in deps[pos_ROOT : pos_Y] and ' that' not in Root:
        MorC = tokens[deps[pos_ROOT : pos_Y].index('mark') + pos_ROOT]
    elif 'cc' in deps[pos_ROOT : pos_Y] and ' that' not in Root:
        MorC = tokens[deps[pos_ROOT : pos_Y].index('cc') + pos_ROOT]
    return MorC


def depParse_n2n(nlp, sentence):
    doc = nlp(sentence)
    tokens = [str(token) for token in doc]
    lemmas = [token.lemma_ for token in doc]
    deps = [token.dep_ for token in doc]
    try:
        pos_X = tokens.index('X')
        pos_ROOT = deps.index('ROOT')
        pos_Y = tokens.index('Y')
        Root = getRoot_n2n(tokens, lemmas, deps)
        MorC = getMarkorConj_n2n(Root, tokens, deps)
        if MorC:
            parse_res = ' '.join([Root, MorC])
        else:
            parse_res = Root
        if len(parse_res.split(' ')) == 1:
            parse_res = parse_res + ' that'
        if pos_X < pos_ROOT and pos_ROOT < pos_Y:
            return True, parse_res
        else:
            return False, None
    except:
        return False, None 


def collect_template_mining_n2n(args):
    nlp = spacy.load(args.nlp_model)
    neuralcoref.add_to_pipe(nlp)

    data_dir = args.data_dir
    data_path = os.path.join(data_dir, args.input_file)
    output_dir = args.template_dir
    os.makedirs(output_dir, exist_ok=True)

    relations = {}

    m_words = ['he']
    f_words = ['she']

    bracket = re.compile(r'\(.*\)|\{.*\}|\[.*\]')

    with jsonlines.open(data_path, 'r') as readers:
        for line in tqdm(readers, desc='Paragraph'):
            nlp_line = nlp(line['text'])
            for sentence in nlp_line.sents:
                sub = None
                gendered_words = []
                sentence = str(sentence).strip().replace('\n\n', ' ')
                if len(sentence) > 0:
                    sentence = bracket.sub('', sentence)
                    nlp_sentence = nlp(sentence)
                    tokens_str = [str(token) for token in nlp_sentence]
                    tokens_span = [token for token in nlp_sentence]
                    deps = [token.dep_ for token in nlp_sentence]
                    try:
                        sub = tokens_span[deps[:deps.index('ROOT')].index('nsubj')]
                    except:
                        pass
                    for m_word in m_words:
                        if m_word in tokens_str:
                            gendered_words.append(m_word)
                    for f_word in f_words:
                        if f_word in tokens_str:
                            gendered_words.append(f_word)    
                    if sub and len(gendered_words) > 0 and str(sub) not in gendered_words:
                        if nlp_sentence._.has_coref:
                            clusters = nlp_sentence._.coref_clusters
                            for cluster in clusters:
                                if sub in cluster.main:
                                    mention = [str(mention) for mention in cluster.mentions]
                                    overlap = list(set(gendered_words)&set(mention))
                                    if overlap:
                                        mi = ' ' + overlap[0] + ' '
                                        end1 = ' ' + overlap[0] + ','
                                        end2 = ' ' + overlap[0] + '.'
                                        judge, parse_res = depParse_n2n(nlp, sentence.replace(str(cluster.main), 'X').replace(mi, ' Y ').replace(end1, ' Y,').replace(end2, ' Y.'))
                                        if judge:
                                            if parse_res in relations.keys():
                                                relations[parse_res] += 1
                                            else:
                                                relations[parse_res] = 1
    relation = [(relation, relations[relation]) for relation in relations]
    relation = sorted(relation, key=lambda relation: relation[1], reverse=True)

    relations_dir = os.path.join(output_dir, args.template_file)
    relations_file = open(relations_dir, 'w')
    relations_writer = csv.writer(relations_file)
    relations_writer.writerow(['template', 'frequency'])
    for item in relation:
        relations_writer.writerow(['The [X] ' + item[0] + ' [Y]', item[1]])
    relations_file.close()


def collect_template_mining_n2a(args):   
    nlp = spacy.load(args.nlp_model)

    data_dir = args.data_dir
    data_path = os.join(data_dir, args.input_file)
    output_dir = args.template_dir
    os.makedirs(output_dir, exist_ok=True)

    roots = {}
    bracket = re.compile(r'\(.*\)|\{.*\}|\[.*\]|\'|\"')
    with jsonlines.open(data_path, 'r') as readers:
        for line in tqdm(readers, desc='Paragraph'):
            nlp_line = nlp(line['text'])
            for sentence in nlp_line.sents:
                sentence = str(sentence).strip().replace('\n\n', ' ')
                try:
                    sentence = bracket.sub('', sentence).strip()
                    if len(sentence) > 2:
                        root = getRoot_n2a(nlp, sentence)
                        if root and (root in roots.keys()):
                            roots[root] += 1
                        elif root:
                            roots[root] = 1
                except:
                    pass

    relations_dir = os.path.join(output_dir, args.template_file)
    relations_file = open(relations_dir, 'w')
    relations_writer = csv.writer(relations_file)
    relations_writer.writerow(['template', 'frequency'])
    for root in roots:
        relations_writer.writerow(['The [X], who' + root + ', is [Y]', roots[root]])
    relations_file.close()


if __name__ == "__main__":
    args = parse_args_for_context_template_collection()
    select_mining_passage_from_wiki()
    collect_template_mining_n2n(args)