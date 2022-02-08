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


def process_vocab(sentences, data_amount):
    vocab = defaultdict(lambda: -1)
    vocab["<start>"] = 0
    vocab["<end>"] = 1
    vocab_reversed = ["<start>", "<end>"]
    count = 2
    processed_sentences = []
    max_len = 0
    min_len = 10000000
    for idx, line in tqdm(enumerate(sentences)):
        buffer = ""
        last_is_alpha = False
        processed_sentences.append([])
        for character in line:
            if character.isalpha():
                buffer += character
            elif not character.isalpha():
                if vocab[buffer] == -1:
                    # vocab not found
                    vocab[buffer] = count
                    vocab_reversed.append(buffer)
                    count += 1
                if vocab[character] == -1:
                    # vocab not found
                    vocab[character] = count
                    vocab_reversed.append(character)
                    count += 1
                processed_sentences[idx].append(vocab[buffer])
                processed_sentences[idx].append(vocab[character])
                buffer = ""        
        processed_sentences[idx] = torch.tensor(
            processed_sentences[idx], dtype=torch.int64
        )
        max_len = max(max_len, len(processed_sentences[idx]))
        min_len = min(min_len, len(processed_sentences[idx]))
        if idx == data_amount:
            break
    return (
        processed_sentences,
        max_len,
        min_len,
        len(vocab.keys()),
        dict(vocab),
        vocab_reversed,
    )


def generate_dataset_from_txt(data_amount):
    print("LOADING GERMAN SENTENCES")
    with open(os.path.join("Translate_Dataset", "europarl-v7.de-en.de"), "r") as file:
        german_sentences = file.read().split("\n")
    print("LOADING ENGLISH SENTENCES")
    with open(os.path.join("Translate_Dataset", "europarl-v7.de-en.en"), "r") as file:
        english_sentences = file.read().split("\n")
    print("PROCESSING GERMAN SENTENCES")
    (
        german_sentences,
        german_max_len,
        german_min_len,
        german_vocab_len,
        german_vocab,
        german_vocab_reversed,
    ) = process_vocab(german_sentences, data_amount)
    print("PROCESSING ENGLISH SENTENCES")
    (
        english_sentences,
        english_max_len,
        english_min_len,
        english_vocab_len,
        english_vocab,
        english_vocab_reversed,
    ) = process_vocab(english_sentences, data_amount)
    print("SAVING TO DISK")
    torch.save(
        {
            "train_data": german_sentences[: int(len(german_sentences) * 0.8)],
            "test_data": german_sentences[int(len(german_sentences) * 0.2) :],
            "max_len": german_max_len,
            "min_len": german_min_len,
            "vocab_len": german_vocab_len,
            "vocab": german_vocab,
            "vocab_reversed": german_vocab_reversed,
        },
        "German_sentences.pkl",
    )
    torch.save(
        {
            "train_data": english_sentences[: int(len(german_sentences) * 0.8)],
            "test_data": english_sentences[int(len(german_sentences) * 0.2) :],
            "max_len": english_max_len,
            "min_len": english_min_len,
            "vocab_len": english_vocab_len,
            "vocab": english_vocab,
            "vocab_reversed": english_vocab_reversed,
        },
        "English_sentences.pkl",
    )


if __name__ == "__main__":
    generate_dataset_from_txt(data_amount=1000)
