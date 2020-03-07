# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 14:34:37 2019
@author: puranam
"""

import re
import itertools
import numpy as np
import pandas as pd
from collections import Counter

"""
Adapted from https://github.com/dennybritz/cnn-text-classification-tf
"""

def clean_str(x):
    x= x.replace("}","")
    x= x.replace("{","")
    x= x.replace(":-)","smiley")
    x= x.replace("{","")
    x= x.replace(")","")
    x= x.replace("(","")
    x= x.replace("."," ")
    x= x.replace("\n"," ")
    new = [i for i in x.strip().lower().split(" ")]
    
    return new

def load_data_and_labels(path):

    x_text, y = load_sentences_and_labels(path)
    #print('text',x_text[0],len(x_text))
    x_text = [s.split(" ")  for s in x_text if len(s)>0]

    y = [[0, 1] if label==1 else [1, 0] for label in y]

    return [x_text, y]

def load_sentences_and_labels(path):
    import pandas as pa
    df = pa.read_csv(path, encoding='latin1')
    df['len']= df['text'].map(lambda x: len(' '.join(x.split(" "))))
    df = df[df['len']>5]
    #print(df.columns,df.shape)
    df['text']= df['text'].map(lambda x: clean_str(x))
    x_text = list(df['text'])#[sent.lower() for sent in list(df['text'])]
    
    return x_text, df['label']

def pad_sentence(sentence, maxlen=False, padding_word="<PAD/>"):

    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """

    sequence_length = maxlen
    num_padding = sequence_length - len(sentence)
        
    if num_padding>0:
        new_sentence = sentence + [padding_word] * num_padding
    else:
        new_sentence = sentence[:sequence_length]

    return new_sentence


def pad_sentences(sentences,maxlen=False, padding_word="<PAD/>"):

    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    if maxlen==False:
        sequence_length = max(len(x) for x in sentences)
    else :
        sequence_length = maxlen

    padded_sentences = []

    for i in range(len(sentences)):

        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        
        if num_padding>0:
            new_sentence = sentence + [padding_word] * num_padding
        else:
            new_sentence = sentence[:sequence_length]

        padded_sentences.append(new_sentence)

    return padded_sentences

def build_vocab(sentences):

    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """

    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))

    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]

    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}

    return vocabulary, vocabulary_inv





def build_input_data(sentences, labels, vocabulary):

    """

    Maps sentencs and labels to vectors based on a vocabulary.

    """
    
    
    a=[]
    for sentence in sentences:
        s=[]
        for word in sentence:
            if word in vocabulary:
                s.append(vocabulary[word])
        a.append(s)
    
    #print("vocabulary[UNK]: ",vocabulary['UNK'])
    x = np.array(a)
    #x= x.astype(int)
    #labels=[int(l) for l in labels]
    #print('max label:  ', max(labels))
    #print('coded:',x[0], x.shape)
    
    y = np.array(labels)
    #print(len(x), len(y))

    y = y.argmax(axis=1)

    return [x, y]



def build_input_eval(sentences, labels, vocabulary, maxlen):

    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    
    a=[]
    for sentence in sentences:

        s=[]
        for word in sentence:

            if word in vocabulary:
                s.append(vocabulary[word])

        s = pad_sentence(s, maxlen, 0)
        a.append(s)
    
    x = np.array(a)
    y = np.array(labels)
    y = y.argmax(axis=1)

    return [x, y]



def load_train(filepath, cv, build_vocabulary=True, maxlen=False):

    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    
    df = pd.read_csv(filepath + '/train_fold' + str(cv) + '.csv', encoding='latin1')
    df['len'] = df['text'].apply(lambda x: len(' '.join(x.split(" "))))
    df = df[df['len']>5]
    df['text'] = df['text'].apply(lambda x: clean_str(x))

    sentences = list(df['text'])
    labels = [[0, 1] if label==1 else [1, 0] for label in list(df['label'])]

    sentences_padded = pad_sentences(sentences, maxlen)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    x, y = build_input_data(sentences_padded, labels, vocabulary)

    return x, y, vocabulary, vocabulary_inv


def load_data(filepath, cv, vocabulary, vocabulary_inv, maxlen):

    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """

    if cv is not None: #Validation set
        df = pd.read_csv(filepath + '/val_fold' + str(cv) + '.csv', encoding='latin1')
    else: #Test set
        df = pd.read_csv(filepath + '/test.csv', encoding='latin1')

    df['len'] = df['text'].apply(lambda x: len(' '.join(x.split(" "))))
    df = df[df['len']>5]
    df['text'] = df['text'].apply(lambda x: clean_str(x))

    sentences = list(df['text'])
    labels = [[0, 1] if label==1 else [1, 0] for label in list(df['label'])]
    x, y = build_input_eval(sentences, labels, vocabulary, maxlen)

    return [x, y, vocabulary, vocabulary_inv]
