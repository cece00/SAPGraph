#!/usr/bin/python
# -*- coding: utf-8 -*-
"""This file contains code to read the train/eval/test data from file and process it, and read the vocab data from file and process it"""

import re
import os
from nltk.corpus import stopwords

import glob
import copy
import random
import time
import json
import pickle
import nltk
import math
import collections
from collections import Counter
from itertools import combinations
import numpy as np
from random import shuffle

import torch
import torch.utils.data
import torch.nn.functional as F

from tools.logger import *

import dgl
from dgl.data.utils import save_graphs, load_graphs


STOPWORD = stopwords.words('english')
punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', '\'\'', '\'', '`', '``', '-', '--', '|', '\/']
STOPWORD.extend(punctuations)


class DatasetItem(object):
    """Class representing a train/val/test example for single-document(paper with hier-secitons) extractive summarization."""

    def __init__(self, article_sents, article_ents, section_names, abstract_sents, vocab, sent_max_len, ent_max_len, label):
        """ Initializes the Example, performing tokenization and truncation to produce the encoder, decoder and target sequences, which are stored in self.

        :param article_sents: list(list(string)) for single-document with sections; one per article sentence. each token is separated by a single space.
        :param article_ents: a 4dim list, doc(sec(sen(ent_3tuple_list)))
        :param abstract_sents: list(strings); one per abstract sentence. In each sentence, each token is separated by a single space.
        :param vocab: Vocabulary object
        :param sent_max_len: int, max length of each sentence
        :param label: list, the No of selected sentence, e.g. [1,3,5]
        """

        self.sent_max_len = sent_max_len
        self.enc_sent_len = []
        self.enc_sent_input = []
        self.enc_sent_input_pad = []

        self.ent_max_len = ent_max_len
        self.ent_input = []
        self.ent_input_pad = []

        # Store the original strings
        self.original_abstract = "\n".join(abstract_sents)
        self.original_article_sents = []
        self.doc_len = [] # to record the length of each section

        self.sec_level_list = []
        self.sec_name_list = []
        self.enc_sec_name = []
        self.enc_sec_name_pad = []

        # Process the article
        if isinstance(article_sents, list) and isinstance(article_sents[0], list):  
            for sec in article_sents:
                self.original_article_sents.extend(sec)    
                self.doc_len.append(len(sec))
        for sent in self.original_article_sents:
            article_words = sent.split()
            self.enc_sent_len.append(len(article_words))  # store the length before padding
            self.enc_sent_input.append([vocab.word2id(w.lower()) for w in article_words])  # list of word ids; OOVs are represented by the id for UNK token
        self._pad_encoder_sent_input(vocab.word2id('[PAD]'))

        # Process the entities
        for sec_ents in article_ents:
            for sen_ents in sec_ents:
                e_ids = []
                e_ids_pad = []
                for item in sen_ents:
                    e = item[2].split()
                    add_ids = [vocab.word2id(w.lower()) for w in e]
                    e_ids.append(add_ids) # entities may be phrases
                    # padding to make sure the same length
                    if len(add_ids) > self.ent_max_len:
                        e_ids_pad.append(add_ids[:self.ent_max_len])
                    else:
                        e_ids_pad.append(add_ids+[vocab.word2id('[PAD]')] * (self.ent_max_len - len(add_ids)))
                self.ent_input.append(e_ids)
                self.ent_input_pad.append(e_ids_pad)
        
        # process section level and section name
        self._get_section_level_nodes(section_names)
        for n in self.sec_name_list:
            split_n = n.split()
            self.enc_sec_name.append([vocab.word2id(w.lower()) for w in split_n])
        self._pad_section_level_nodes(vocab.word2id('[PAD]'))

        # Store the label
        self.label = label
        label_shape = (len(self.original_article_sents), len(label))  # [N, len(label)]
        self.label_matrix = np.zeros(label_shape, dtype=int)
        if label != []:
            self.label_matrix[np.array(label), np.arange(len(label))] = 1  # label_matrix[i][j]=1 indicate the i-th sent will be selected in j-th step


    def _pad_encoder_sent_input(self, pad_id):
        """
        :param pad_id: int; token pad id
        :return: 
        """
        max_len = self.sent_max_len
        for i in range(len(self.enc_sent_input)):
            article_words = self.enc_sent_input[i].copy()
            if len(article_words) > max_len:
                article_words = article_words[:max_len]
            if len(article_words) < max_len:
                article_words.extend([pad_id] * (max_len - len(article_words)))
            self.enc_sent_input_pad.append(article_words)


    def _get_section_level_nodes(self, name_list): 
        """
        # :param sec_name: 2 dim list of section names
        :param name_list: 2 dim list of splited section names
        :return: dict(dict), {"section_id":{"sec_node_id": j, "parent_node": k}...}, normally section_id==sec_node_id (section_id>0), but if section_id<0(such as -1, -2,...). It is just a supernode (with no sentence in it) where sec_node_id !=section_id. if it has no parent node, "parent_node"=0
        """
        
        keyword_A = ['intro', 'purpose', 'background']
        keyword_B = ['design', 'method', 'approach']
        keyword_C = ['result', 'find', 'discuss', 'analy']
        keyword_D = ['conclu', 'future']
        keywords = [keyword_A, keyword_B, keyword_C, keyword_D]

        def sec_cls(n_str): # return {0,1,2,3,4}
            res_list = [0,0,0,0]
            for kw_i in range(len(keywords)):
                for w in keywords[kw_i]:
                    if n_str.find(w) != -1:
                        res_list[kw_i] += 1 # vote: if found keyword_i, then add 1 to keyword_i's index
            if sum(res_list) == 0:
                return 4
            else:
                return np.argmax(res_list)   

        super_id = -1
        normal_id = 1
        sec_i = {}
        for i in range(len(name_list)):
            n_tmp = name_list[i]
            if n_tmp:
                if len(n_tmp) == 1: # it's a 1level section
                    cls = sec_cls(n_tmp[0])
                    sec_i[n_tmp[0]] = {"sec_node_id": normal_id, "parent_node": 0, "sec_cls":cls}
                    normal_id +=1

                elif len(n_tmp)==2: # it's a 2level section
                    cls = sec_cls(' '.join(n_tmp))
                    if n_tmp[-1] not in sec_i.keys(): 
                        sec_i[n_tmp[-1]] = {"sec_node_id": super_id, "parent_node": 0, "sec_cls":cls}
                        super_id -= 1
                    if n_tmp[0] not in sec_i.keys():
                        sec_i[n_tmp[0]] = {"sec_node_id": normal_id, "parent_node": sec_i[n_tmp[-1]]["sec_node_id"], "sec_cls":cls}
                        normal_id+=1

                elif len(n_tmp)>=3: # it's a >=3level section, >=3level must not be a super node
                    cls = sec_cls(' '.join(n_tmp))
                    if n_tmp[-1] not in sec_i.keys(): 
                        sec_i[n_tmp[-1]] = {"sec_node_id": super_id, "parent_node": 0, "sec_cls":cls}
                        super_id -= 1
                    if n_tmp[-2] not in sec_i.keys(): 
                        sec_i[n_tmp[-2]] = {"sec_node_id": super_id, "parent_node": sec_i[n_tmp[-1]]["sec_node_id"], "sec_cls":cls}
                        super_id -= 1
                    if n_tmp[-3] not in sec_i.keys():  # change here if need all levels
                        sec_i[' '.join(n_tmp[:-2][::-1])] = {"sec_node_id": normal_id, "parent_node": sec_i[n_tmp[-2]]["sec_node_id"], "sec_cls":cls}
                        normal_id +=1
            else:
                print('the neme_list_i is none ', i)
        
        for k,v in sec_i.items():
            self.sec_name_list.append(k)
            self.sec_level_list.append([v['sec_node_id'], v['parent_node'], v['sec_cls']])
        assert len(self.sec_name_list)==len(sec_i), "sec_level_length_error"


    def _pad_section_level_nodes(self, pad_id):
        """
        :param pad_id: int; token pad id
        :return: 
        """
        max_len = self.sent_max_len
        for i in range(len(self.enc_sec_name)):
            article_words = self.enc_sec_name[i].copy()
            if len(article_words) > max_len:
                article_words = article_words[:max_len]
            if len(article_words) < max_len:
                article_words.extend([pad_id] * (max_len - len(article_words)))
            self.enc_sec_name_pad.append(article_words)




def graph_collate_fn(samples):
    '''
    :param batch: (G, input_pad)
    :return: 
    '''
    graphs, index = map(list, zip(*samples))
    graph_len = [len(g.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)) for g in graphs]  # sent node of graph
    sorted_len, sorted_index = torch.sort(torch.LongTensor(graph_len), dim=0, descending=True)
    batched_graph = dgl.batch([graphs[idx] for idx in sorted_index])
    return batched_graph, [index[idx] for idx in sorted_index]
