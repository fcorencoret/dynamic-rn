import json
import os
import pickle
import re

import torch
from tqdm import tqdm

classes = {
            'number':['0','1','2','3','4','5','6','7','8','9','10'],
            'material':['rubber','metal'],
            'color':['cyan','blue','yellow','purple','red','green','gray','brown'],
            'shape':['sphere','cube','cylinder'],
            'size':['large','small'],
            'exist':['yes','no']
        }

def build_dictionaries(clevr_dir):

    def compute_class(answer):
        for name,values in classes.items():
            if answer in values:
                return name
        
        raise ValueError('Answer {} does not belong to a known class'.format(answer))
        
        
    cached_dictionaries = os.path.join('clevr-humans', 'CLEVR_built_dictionaries.pkl')
    if os.path.exists(cached_dictionaries):
        print('==> using cached dictionaries: {}'.format(cached_dictionaries))
        with open(cached_dictionaries, 'rb') as f:
            return pickle.load(f)
            
    quest_to_ix = {}
    quest_count = {}
    answ_to_ix = {}
    answ_ix_to_class = {}
    json_train_filename = os.path.abspath('clevr-humans/CLEVR-Humans-train.json')
    #load all words from all training data
    questions = []
    with open(json_train_filename, "r") as f:
        questions += json.load(f)['questions']

    json_val_filename = os.path.abspath('clevr-humans/CLEVR-Humans-val.json')
    #load all words from all training data
    with open(json_val_filename, "r") as f:
        questions += json.load(f)['questions']
   
    for q in tqdm(questions):
        question = tokenize(q['question'])
        answer = q['answer']
        #pdb.set_trace()
        for word in question:
            if word not in quest_count:
                # quest_to_ix[word] = len(quest_to_ix)+1 #one based indexing; zero is reserved for padding
                quest_count[word] = 1
            else:
                quest_count[word] += 1
        
        a = answer.lower()
        if a not in answ_to_ix:
                ix = len(answ_to_ix)+1
                answ_to_ix[a] = ix
                answ_ix_to_class[ix] = compute_class(a)


    for q  in tqdm(questions):
        question = tokenize(q['question'])
        for word in question:
            if quest_count[word] >= 10:
                quest_to_ix[word] = len(quest_to_ix)+1 #one based indexing; zero is reserved for padding

    unkown_word_index = len(quest_to_ix) + 1

    for q  in tqdm(questions):
        question = tokenize(q['question'])
        for word in question:
            if quest_count[word] < 10:
                quest_to_ix[word] = unkown_word_index

    # json_test_filename = os.path.abspath('clevr-humans/CLEVR-Humans-test.json')
    # #load all words from all training data
    # with open(json_test_filename, "r") as f:
    #     questions = json.load(f)['questions']
    #     for q in tqdm(questions):
    #         question = tokenize(q['question'])
    #         answer = q['answer']
    #         #pdb.set_trace()
    #         for word in question:
    #             if word not in quest_to_ix:
    #                 quest_to_ix[word] = len(quest_to_ix)+1 #one based indexing; zero is reserved for padding
            
    #         a = answer.lower()
    #         if a not in answ_to_ix:
    #                 ix = len(answ_to_ix)+1
    #                 answ_to_ix[a] = ix
    #                 answ_ix_to_class[ix] = compute_class(a)
    
    ret = (quest_to_ix, answ_to_ix, answ_ix_to_class)    
    with open(cached_dictionaries, 'wb') as f:
        pickle.dump(ret, f)

    return ret


def to_dictionary_indexes(dictionary, sentence):
    """
    Outputs indexes of the dictionary corresponding to the words in the sequence.
    Case insensitive.
    """
    split = tokenize(sentence)
    idxs = torch.LongTensor([dictionary[w] for w in split])
    return idxs

def collate_samples_from_pixels(batch):
    return collate_samples(batch, False, False)
    
def collate_samples_state_description(batch):
    return collate_samples(batch, True, False)

def collate_samples_images_state_description(batch):
    return collate_samples(batch, True, True)
    
def collate_samples(batch, state_description, only_images):
    """
    Used by DatasetLoader to merge together multiple samples into one mini-batch.
    """
    batch_size = len(batch)

    if only_images:
        images = batch
    else:
        images = [d['image'] for d in batch]
        answers = [d['answer'] for d in batch]
        questions = [d['question'] for d in batch]

        # questions are not fixed length: they must be padded to the maximum length
        # in this batch, in order to be inserted in a tensor
        max_len = max(map(len, questions))

        padded_questions = torch.LongTensor(batch_size, max_len).zero_()
        for i, q in enumerate(questions):
            padded_questions[i, :len(q)] = q
        
    if state_description:
        max_len = 12
        #even object matrices should be padded (they are variable length)
        padded_objects = torch.FloatTensor(batch_size, max_len, images[0].size()[1]).zero_()
        for i, o in enumerate(images):
            padded_objects[i, :o.size()[0], :] = o
        images = padded_objects
    
    if only_images:
        collated_batch = torch.stack(images)
    else:
        collated_batch = dict(
            image=torch.stack(images),
            answer=torch.stack(answers),
            question=padded_questions
        )
    return collated_batch


def tokenize(sentence):
    # punctuation should be separated from the words
    s = re.sub('([.,;:!?()])', r' \1 ', sentence)
    s = re.sub('\s{2,}', ' ', s)

    # tokenize
    split = s.split()

    # normalize all words to lowercase
    lower = [w.lower() for w in split]
    return lower


def load_tensor_data(data_batch, cuda, invert_questions, volatile=False):
    # prepare input
    # var_kwargs = dict(volatile=True) if volatile else dict(requires_grad=False)

    qst = data_batch['question']
    if invert_questions:
        # invert question indexes in this batch
        qst_len = qst.size()[1]
        qst = qst.index_select(1, torch.arange(qst_len - 1, -1, -1).long())

    img = data_batch['image'].clone()
    qst = qst.clone()
    label = data_batch['answer'].clone()
    if cuda:
        img, qst, label = img.cuda(), qst.cuda(), label.cuda()

    label = (label - 1).squeeze(1)
    return img, qst, label
