import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import random
import time

def process_vocab(sentences,data_amount):
    vocab = defaultdict(lambda:-1)
    vocab["<start>"] = 0
    vocab["<end>"] = 1    
    vocab_reversed = ["<start>","<end>"]
    count = 2
    processed_sentences = []
    max_len = 0
    min_len = 10000000
    for idx,line in tqdm(enumerate(sentences)):
        buffer = ""
        last_is_alpha = False
        processed_sentences.append([])    
        for character in line:
            if(character.isalpha()):
                buffer += character
            elif(not character.isalpha()):
                if(vocab[buffer]==-1):
                    #vocab not found
                    vocab[buffer] = count                        
                    vocab_reversed.append(buffer)
                    count +=1
                if(vocab[character]==-1):
                    #vocab not found
                    vocab[character] = count
                    vocab_reversed.append(character)
                    count +=1
                processed_sentences[idx].append(vocab[buffer])
                processed_sentences[idx].append(vocab[character])
                buffer = ""        
        processed_sentences[idx].append(vocab["<end>"])
        processed_sentences[idx] = torch.tensor(processed_sentences[idx],dtype=torch.int64)
        max_len = max(max_len,len(processed_sentences[idx]))
        min_len = min(min_len,len(processed_sentences[idx]))
        if(idx==data_amount):
            break
    return processed_sentences,max_len,min_len,len(vocab.keys()),dict(vocab),vocab_reversed
def generate_dataset_from_txt(data_amount):
    print("LOADING GERMAN SENTENCES")
    with open(os.path.join("Translate_Dataset","europarl-v7.de-en.de"),"r") as file:
        german_sentences = file.read().split("\n")            
    print("LOADING ENGLISH SENTENCES")
    with open(os.path.join("Translate_Dataset","europarl-v7.de-en.en"),"r") as file:
        english_sentences = file.read().split("\n")
    print("PROCESSING GERMAN SENTENCES")
    german_sentences,german_max_len,german_min_len,german_vocab_len,german_vocab,german_vocab_reversed = process_vocab(german_sentences,data_amount)
    print("PROCESSING ENGLISH SENTENCES")
    english_sentences,english_max_len,english_min_len,english_vocab_len,english_vocab,english_vocab_reversed = process_vocab(english_sentences,data_amount)
    print("SAVING TO DISK")
    torch.save(
        {"train_data":german_sentences[:int(len(german_sentences)*0.8)],
        "test_data":german_sentences[int(len(german_sentences)*0.2):],
        "max_len":german_max_len,
        "min_len":german_min_len,
        "vocab_len":german_vocab_len,
        "vocab":german_vocab,
        "vocab_reversed":german_vocab_reversed},"German_sentences.pkl")
    torch.save(
        {"train_data":english_sentences[:int(len(german_sentences)*0.8)],
        "test_data":english_sentences[int(len(german_sentences)*0.2):],
        "max_len":english_max_len,
        "min_len":english_min_len,
        "vocab_len":english_vocab_len,
        "vocab":english_vocab,
        "vocab_reversed":english_vocab_reversed},"English_sentences.pkl")

class EnglishToGermanDataset(torch.utils.data.Dataset):
    def __init__(self,CUDA=False):
        super(EnglishToGermanDataset, self).__init__()                
        print("LOADING GERMAN SENTENCES")
        load = torch.load("German_sentences_full.pkl")
        self.german_sentences_train = load["train_data"]
        self.german_sentences_test = load["test_data"]
        self.german_max_len = load["max_len"]
        self.german_min_len = load["min_len"]
        self.german_vocab_len = load["vocab_len"]
        self.german_vocab = load["vocab"]
        self.german_vocab_reversed = load["vocab_reversed"]
        self.german_eos = self.german_vocab["<end>"]
        print("LOADING ENGLISH SENTENCES")
        load = torch.load("English_sentences_full.pkl")
        self.english_sentences_train = load["train_data"]
        self.english_sentences_test = load["test_data"]
        self.english_max_len = load["max_len"]    
        self.english_min_len = load["min_len"]    
        self.english_vocab_len = load["vocab_len"]
        self.english_vocab = load["vocab"]
        self.english_vocab_reversed = load["vocab_reversed"]
        self.mode = "train"
        self.english_eos = self.english_vocab["<end>"]
        self.min_len = 30#min(self.german_min_len,self.english_min_len)
        self.CUDA = CUDA
        self.device = torch.device('cuda:0' if CUDA else 'cpu')
    def logit_to_sentence(self,logits,language="german"):
        if(language=="german"):
            vocab = self.german_vocab_reversed
        else:
            vocab = self.english_vocab_reversed
        sentence = []
        for l in logits:
            idx = torch.argmax(l)
            word = vocab[idx]
            sentence.append(word)
        return "".join(sentence)
    def test(self):
        self.mode = "test"
    def train(self):
        self.mode = "train"
    def __getitem__(self, idx):              
        torch.set_default_tensor_type(torch.FloatTensor)  
        if(self.mode=="test"):
            german_item = self.german_sentences_test[idx]
            english_item = self.english_sentences_test[idx]
        else:
            german_item = self.german_sentences_train[idx]
            english_item = self.english_sentences_train[idx]
        min_len = min(len(german_item),len(english_item))        
        start_token = torch.tensor([self.german_vocab["<start>"]],dtype=torch.int64)        
        if(min_len>self.min_len):
            #Crop randomly
            crop_range = min(len(german_item),len(english_item)) - self.min_len
            crop = random.randint(0,crop_range)
            german_item = german_item[crop:self.min_len+crop]
            english_item = english_item[crop:self.min_len+crop]
            german_item = torch.cat((start_token,german_item))
            logit_mask = torch.ones((len(german_item),1),dtype=torch.bool)
        else:
            german_item = F.pad(german_item,(0,self.min_len-len(german_item)),"constant",self.german_eos)
            english_item = F.pad(english_item,(0,self.min_len-len(english_item)),"constant",self.english_eos)
            german_item = torch.cat((start_token,german_item))
            #Logit Mask For Training
            logit_mask = torch.ones((len(german_item),1),dtype=torch.bool)
            logit_mask[min_len+1:,:] = 0
        german_logits = torch.zeros((len(german_item),self.german_vocab_len))
        index = torch.arange(0,len(german_item))
        german_logits[index,german_item] = 1                
        torch.set_default_tensor_type(torch.cuda.FloatTensor)  
        return {"german":german_item.to(self.device),
                "english":english_item.to(self.device),
                "logits":german_logits.to(self.device),
                "logit_mask":logit_mask.to(self.device)}
    def __len__(self):
        if(self.mode=="test"):
            return len(self.german_sentences_test)
        else:
            return len(self.german_sentences_train)
if __name__=="__main__":
    generate_dataset_from_txt(data_amount=1000)