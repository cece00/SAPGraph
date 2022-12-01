#!/usr/bin/python
# -*- coding: utf-8 -*-
"""This file contains code to gather data from dataloader_item.py, and construct graph"""

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
from module_ffn.dataloader_item import *

import dgl
from dgl.data.utils import save_graphs, load_graphs

import gzip

STOPWORD = stopwords.words('english')
punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', '\'\'', '\'', '`', '``', '-', '--', '|', '\/']
STOPWORD.extend(punctuations)

impo_weight = {0:0.35, 1:0.1, 2:0.15, 3:0.35, 4:0.05} 


class DatasetAll(torch.utils.data.IterableDataset):
    """ Constructor: Dataset of DatasetItem(object) for single document summarization"""

    def __init__(self, data_path, vocab, doc_max_timesteps, sent_max_len, ent_max_len, filter_word_path, w2s_path, s2s_path, thres_sim):
        """ Initializes the DatasetAll with the path of data
        
        :param data_path: string; the path of data
        :param vocab: object;
        :param doc_max_timesteps: int; the maximum sentence number of a document, each example should pad sentences to this length
        :param sent_max_len: int; the maximum token number of a sentence, each sentence should pad tokens to this length
        :param ent_max_len: int; the maximum token number of an entity, each entity should pad tokens to this length
        :param filter_word_path: str; file path, the file must contain one word for each line and the tfidf value must go from low to high (the format can refer to script/lowTFIDFWords.py) 
        :param w2s_path: str; file path, each line in the file contain a json format data (which can refer to the format can refer to script/calw2sTFIDF.py)
        :param s2s_path:
        :param thres_sim:
        """

        self.vocab = vocab
        self.sent_max_len = sent_max_len
        self.ent_max_len = ent_max_len
        self.doc_max_timesteps = doc_max_timesteps
        self.data_path = data_path
        
        logger.info("[INFO] Loading entity2sent TFIDF file from %s!" % w2s_path)
        self.w2s_path = w2s_path
        logger.info("[INFO] Loading sen2sen TFIDF file from %s!" % s2s_path)
        self.s2s_path = s2s_path

        self.thres_sim = thres_sim

        logger.info("[INFO] Start reading %s", self.__class__.__name__)
        # Need to set overall data size, because of iteration, TODO: add to param
        self.size = 85908
        self.start = 0
        self.end = self.size

        logger.info("[INFO] Loading filter word File %s", filter_word_path)
        tfidf_w = readText(filter_word_path)
        self.filterwords = STOPWORD
        self.filterids = [vocab.word2id(w.lower()) for w in STOPWORD]
        self.filterids.append(vocab.word2id("[PAD]"))   # keep "[UNK]" but remove "[PAD]"
        lowtfidf_num = 0
        # pattern = r"^[0-9]+$"
        for w in tfidf_w:
            if vocab.word2id(w) != vocab.word2id('[UNK]'):
                self.filterwords.append(w)
                self.filterids.append(vocab.word2id(w))
                # if re.search(pattern, w) == None:  # if w is a number, it will not increase the lowtfidf_num
                #     lowtfidf_num += 1
                lowtfidf_num += 1
            if lowtfidf_num > 5000: # out the 5000 lowest tfidf word
                break

    
    def pad_label_m(self, label_matrix):
        label_m = label_matrix[:self.doc_max_timesteps, :self.doc_max_timesteps]
        N, m = label_m.shape
        if m < self.doc_max_timesteps:
            pad_m = np.zeros((N, self.doc_max_timesteps - m))
            return np.hstack([label_m, pad_m])
        return label_m


    def AddEntNode(self, G, ent_id):
        tid2nid = [] # list is not hashable
        nid2tid = {}
        nid = 0       
        for sen in ent_id:
            for item in sen:
                out = []
                for wid in item:
                    if wid not in self.filterids or wid == self.vocab.word2id("[PAD]"):
                        out.append(1)
                    else:
                        out.append(0)
                if sum(out) == len(out) and item not in tid2nid:
                    tid2nid.append(item)
                    nid2tid[nid] = item
                    nid += 1
        
        t_nodes = len(nid2tid)
        if t_nodes!=0:
            G.add_nodes(t_nodes)
            G.set_n_initializer(dgl.init.zero_initializer)
            G.ndata["unit"] = torch.zeros(t_nodes)
            G.ndata["ids"] = torch.LongTensor(list(nid2tid.values())) # [[1, 2], [3, 4], [5, 6]]
            G.ndata["dtype"] = torch.zeros(t_nodes)
        else:
            print('ent_id :  ',ent_id)
            if ent_id and ent_id[0]:
                tid2nid.append(ent_id[0][0])
                nid2tid[nid] = ent_id[0][0]
            else:
                tid2nid.append([self.vocab.word2id("[PAD]")]*3)
                nid2tid[nid] = [self.vocab.word2id("[PAD]")]*3
                
            t_nodes = 1
            G.add_nodes(t_nodes)
            G.set_n_initializer(dgl.init.zero_initializer)
            G.ndata["unit"] = torch.zeros(t_nodes)
            G.ndata["ids"] = torch.LongTensor(list(nid2tid.values())) # [[1, 2], [3, 4], [5, 6]]
            G.ndata["dtype"] = torch.zeros(t_nodes)

        return tid2nid, nid2tid

    def MapSent2sec(self, article_len, sentNum):
        # print("in MapSent2sec: ",article_len, sentNum)
        sent2sec = {}
        sec2sent = {}
        sentNo = 0
        for i in range(len(article_len)):
            sec2sent[i] = []
            for j in range(article_len[i]):
                sent2sec[sentNo] = i
                sec2sent[i].append(sentNo)
                sentNo += 1
                if sentNo >= sentNum:
                    return sent2sec
        return sent2sec

    def CreateGraph(self, docLen, ent_pad, sent_pad, sec_level, sec_name_pad, label, w2s_w, s2s_w, thres_sim):
        """ Create a graph for each document

        :param docLen: list; the length of each document in this example
        :param sent_pad: list(list), [sentnum, wordnum] # list（每个二维list应该是[32，213，54...]这种word id list）
        :param doc_pad: list, [document, wordnum]
        :param label: list(list), [sentnum, sentnum]
        :param w2s_w: dict(dict) {str: {str: float}}, for each sentence and each word, the tfidf between them
        :param w2d_w: dict(dict) {str: {str: float}}, for each document and each word, the tfidf between them
        :return: G: dgl.DGLGraph
            node:
                entity: unit=0, dtype=0, id=(int)wordid in vocab
                sentence: unit=1, dtype=1, words=tensor, position=int, label=tensor
                sections: unit=1, dtype=2, words=tensor
            edge:
                ent2sent, sent2ent: tffrac=int, dtype=0
                sent2sent: tffrac=int, dtype=1
                sent2sec, sec2sent: tffrac=ones, dtype=2
                sec2sec: tffrac=ones, dtype=3
        """
        # add word nodes
        G = dgl.DGLGraph()
        assert len(ent_pad)!=0, 'len(ent_pad) == 0'
        tid2nid, nid2tid = self.AddEntNode(G, ent_pad)
        t_nodes = len(nid2tid)
        
        # assert t_nodes!=0, 't_nodes == 0'

        # add sent nodes
        N = len(sent_pad) # 每组sent的数量
        G.add_nodes(N)
        G.ndata["unit"][t_nodes:] = torch.ones(N)
        G.ndata["dtype"][t_nodes:] = torch.ones(N)
        sentid2nid = [i + t_nodes for i in range(N)]
        tsen_nodes = t_nodes + N

        # add section nodes
        sent2sec = self.MapSent2sec(docLen, N)
        sec_num = len(sec_name_pad)
        G.add_nodes(sec_num)

        norsec_ids = []
        topsec_ids = []
        for i in range(len(sec_level)):
            if sec_level[i][1] == 0:
                topsec_ids.append(i)
            if sec_level[i][0]>0:
                norsec_ids.append(i)
                
        G.ndata["unit"][tsen_nodes:] = torch.ones(sec_num)
        G.ndata["dtype"][tsen_nodes:] = torch.ones(sec_num) * 2 # tensor[2,2,2...]
        for top_i in topsec_ids:
            if tsen_nodes+top_i<G.ndata["dtype"].shape[0]:
                G.ndata["dtype"][tsen_nodes+top_i] = torch.tensor([3])

        # G.ndata["dtype"][tsen_nodes:tsen_nodes+topsec_num] = torch.ones(topsec_num) * 2        
        # G.ndata["dtype"][tsen_nodes+topsec_num:] = torch.ones(sec_num-topsec_num) * 3
        secid2nid = [i + tsen_nodes for i in range(sec_num)]

        # map sent and sec on node_id level, 存储该句和属于它的sec的距离，因为batch之后图节点编号会改变
        map_nid_sensec = [secid2nid[sent2sec[i]]-sentid2nid[i] for i in range(len(sent2sec))]
        map_nid_sensec += ([0]*(max(N-len(sent2sec),0)))

        # add entity to sent edges
        # tid2nid_set = sum(tid2nid, [])
        for i in range(len(sent2sec)):
            # c = Counter(sent_pad[i])
            sent_nid = sentid2nid[i]
            sent_tfw = w2s_w[str(i)]
            tfidf_ent = 0
            for enti in tid2nid:
                tfidf_ent = 0
                add_e = False
                for wid in enti:
                    if wid in sent_pad[i] and self.vocab.id2word(wid) in sent_tfw.keys():
                        tfidf_ent += sent_tfw[self.vocab.id2word(wid)]
                        add_e = True
                    else:
                        if wid == self.vocab.word2id('[PAD]'):
                            pass
                        else:
                            tfidf_ent = 0
                            add_e = False
                if add_e:
                    tfidf = tfidf_ent/self.ent_max_len
                    tfidf_box = np.round(tfidf * 9)  # box = 10

                    # w2s s2w
                    G.add_edges(tid2nid.index(enti), sent_nid,
                               data={"tffrac": torch.LongTensor([tfidf_box]), "dtype": torch.Tensor([0])})
                    G.add_edges(sent_nid, tid2nid.index(enti),
                               data={"tffrac": torch.LongTensor([tfidf_box]), "dtype": torch.Tensor([0])})
            # add sent to normal sec_name
            secid = sent2sec[i]
            if secid < len(norsec_ids):
                norsecid = norsec_ids[secid]
                
                if norsecid>=len(secid2nid):
                    # print('norsec_ids,secid2nid: ',norsec_ids,secid2nid,norsecid)
                    continue
                secnid = secid2nid[norsecid]

                '''
                # add all edge weight as 1
                G.add_edges(sent_nid, secnid, data={"tffrac": torch.LongTensor([1]), "dtype": torch.Tensor([2])})
                G.add_edges(secnid, sent_nid, data={"tffrac": torch.LongTensor([1]), "dtype": torch.Tensor([2])})
                '''
                # add edge weight as sent pos
                sent_num = docLen[secid]
                sent_relative_pos = i-sum(docLen[:secid])
                e_w = 1-2*min(sent_relative_pos,sent_num-sent_relative_pos)/sent_num
                imp_w = impo_weight[ sec_level[norsecid][2] ]
                # pos_w = max(i+1,N-i-1)/N

                e_w_box = np.round(e_w * imp_w * 9)
                # e_w_box = np.round(imp_w * 9)
                G.add_edges(sent_nid, secnid, data={"tffrac": torch.LongTensor([e_w_box]), "dtype": torch.Tensor([2])})
                G.add_edges(secnid, sent_nid, data={"tffrac": torch.LongTensor([e_w_box]), "dtype": torch.Tensor([2])})
            else:
                # logger.info("[WARNING] there is no such normal section %d ", secid)
                pass


        # add sent to sent edges
        offset,i,j = 0,0,0
        is_over = False
        # print("docLen ", docLen)
        for s in range(len(docLen)):
            sim_mat = s2s_w[str(s)]
            # print('sim_mat ',len(sim_mat))
            if is_over:
                break
            for i in range(len(sim_mat)):
                if i+offset >= self.doc_max_timesteps:
                    is_over = True
                    break
                for j in range(i, len(sim_mat)):
                    if j+offset >= self.doc_max_timesteps:
                        break
                    if sim_mat[i][j]> thres_sim:
                        # sim_box = np.round(sim_mat[i][j] * 9, 1)
                        sim_box = np.round(sim_mat[i][j] * 9)
                        G.add_edges(sentid2nid[i+offset], sentid2nid[j+offset],
                               data={"tffrac": torch.LongTensor([sim_box]), "dtype": torch.Tensor([1])})
                        G.add_edges(sentid2nid[j+offset], sentid2nid[i+offset],
                               data={"tffrac": torch.LongTensor([sim_box]), "dtype": torch.Tensor([1])})
                    
            offset = sum(docLen[:s])
            # print("offset ", s, offset, docLen[s])
        # add sent data
        G.nodes[sentid2nid].data["words"] = torch.LongTensor(sent_pad)  # [N, seq_len]
        G.nodes[sentid2nid].data["position"] = torch.arange(1, N + 1).view(-1, 1).long()  # [N, 1]
        G.nodes[sentid2nid].data["label"] = torch.LongTensor(label)  # [N, doc_max]
        G.nodes[sentid2nid].data['sec_nid_offset'] = torch.LongTensor(map_nid_sensec)

        # print('torch.LongTensor(label) ',torch.LongTensor(label).size())
        # print("sent num: ",len(sentid2nid))
        # print("sec num: ",sec_num)

        # add normal sec to super sec edges 
        # calculate section level tree to get position, to get edge weight
        level_dic = {}
        for i in range(len(sec_level)):
            sec_id = sec_level[i][0]
            parent_id = sec_level[i][1]
            if parent_id not in level_dic.keys():
                level_dic[parent_id] = [sec_id]
            else:
                level_dic[parent_id].append(sec_id)

        # print('docLen, norsec_ids: ',docLen, norsec_ids,'\n', sec_level,'\n', level_dic,'\n',sec_name_pad,'\n')


        # print('sec_level ', sec_level)
        # print('level_dic ', level_dic)
        for i in range(len(sec_level)):
            sec_id = sec_level[i][0]
            parent_id = sec_level[i][1]
            if parent_id == 0:
                pass
            else:
                # print('parent_id ', parent_id)
                try:
                    parent_ind = [j for j in range(len(sec_level)) if sec_level[j][0] == parent_id][0]
                    # print('parent_ind ',parent_ind)
                    '''
                    # add all edge weight as 1
                    G.add_edges(secid2nid[i], secid2nid[parent_ind],
                                data={"tffrac": torch.LongTensor([1]), "dtype": torch.Tensor([3])})
                    G.add_edges(secid2nid[parent_ind], secid2nid[i],
                                data={"tffrac": torch.LongTensor([1]), "dtype": torch.Tensor([3])})
                    '''
                    # add edge weight as sec pos
                    sec_relative_pos = level_dic[parent_id].index(sec_id)
                    
                    sec_e_w = 1-2*min(sec_relative_pos,len(level_dic[parent_id])-sec_relative_pos)/len(level_dic[parent_id])
                    sec_e_w_box = np.round(sec_e_w * 9)
                    G.add_edges(secid2nid[i], secid2nid[parent_ind],
                                data={"tffrac": torch.LongTensor([sec_e_w_box]), "dtype": torch.Tensor([3])})
                    G.add_edges(secid2nid[parent_ind], secid2nid[i],
                                data={"tffrac": torch.LongTensor([sec_e_w_box]), "dtype": torch.Tensor([3])})
                except:
                    # print('was cut')
                    pass
                               
        # add super sec to super sec (just for experiment)
        if len(topsec_ids):
            topsec_ids.sort()
            for i in range(len(topsec_ids)-1):
                if topsec_ids[i] >= len(secid2nid) or topsec_ids[i+1]>=len(secid2nid):
                    continue
                G.add_edges(secid2nid[topsec_ids[i]], secid2nid[topsec_ids[i+1]],
                               data={"tffrac": torch.LongTensor([1]), "dtype": torch.Tensor([3])})
                G.add_edges(secid2nid[topsec_ids[i+1]], secid2nid[topsec_ids[i]],
                               data={"tffrac": torch.LongTensor([1]), "dtype": torch.Tensor([3])})
        # add sec data
        if sec_num !=0:
            G.nodes[secid2nid].data['words'] = torch.LongTensor(sec_name_pad)
            G.nodes[secid2nid].data["label"] = torch.LongTensor([[0]*self.doc_max_timesteps]*sec_num)  # [N, doc_max]

        # print('G.num_nodes ',G.num_nodes())
        
        return G


    def _yield_data(self):
#         with gzip.open(self.data_path) as f:
#             for line in f:
#                 json_str = line.decode('utf-8')            # 2. string (i.e. JSON)
#                 yield json.loads(json_str)
        with open(self.data_path, encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)
    
    def _yield_w2s_tfidf(self):
#         with gzip.open(self.w2s_path) as f:
#              for line in f:
#                 json_str = line.decode('utf-8')            # 2. string (i.e. JSON)
#                 yield json.loads(json_str)
        with open(self.w2s_path, encoding="utf-8") as f:
             for line in f:
                 yield json.loads(line)
                
                
    def _yield_s2s_tfidf(self):
#         with gzip.open(self.s2s_path) as f:
#              for line in f:
#                 json_str = line.decode('utf-8')            # 2. string (i.e. JSON)
#                 yield json.loads(json_str)
        with open(self.s2s_path, encoding="utf-8") as f:
             for line in f:
                 yield json.loads(line)
                
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        
        if worker_info is None:  # single-process data loading, return the full iterator
            print("w_info is none, sorry")
            iter_start = self.start
            iter_end = self.end
        else:
            print("w info id: ",worker_info.id)
            per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = self.start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.end)
        for index, (data, w2s, s2s) in enumerate(zip(self._yield_data(), self._yield_w2s_tfidf(), self._yield_s2s_tfidf())):
            if index >= iter_start and index < iter_end:
                data["summary"] = data.setdefault("summary", [])
                # (self, article_sents, article_ents, section_names, abstract_sents, vocab, sent_max_len, label)
                
                example = DatasetItem(data["text"],data["entity"], data["section_name"], data["summary"], self.vocab, self.sent_max_len, self.ent_max_len, data['label'])

                ent_pad = example.ent_input_pad[:self.doc_max_timesteps]
                sent_pad = example.enc_sent_input_pad[:self.doc_max_timesteps]
                doc_len = example.doc_len

                now_l = 0
                nor_sec_max_timesteps = 0
                for s_l in range(len(doc_len)):
                    now_l += doc_len[s_l]
                    nor_sec_max_timesteps+=1
                    if now_l>self.doc_max_timesteps:
                        doc_len[s_l] = self.doc_max_timesteps - sum(doc_len[:s_l])
                        break
                sec_max_timesteps = 0
                j = 0
                for sl in example.sec_level_list:
                    if sl[0]>0:
                        j+=1
                    sec_max_timesteps+=1
                    if j == nor_sec_max_timesteps:
                        break

                sec_level = example.sec_level_list[:sec_max_timesteps]
                sec_name_pad = example.enc_sent_input_pad[:sec_max_timesteps]
                doc_len = example.doc_len[:sec_max_timesteps]

                label = self.pad_label_m(example.label_matrix)
                
                G = self.CreateGraph(doc_len,ent_pad, sent_pad, sec_level,sec_name_pad, label, w2s, s2s, self.thres_sim)
                yield G, index
            else:
                continue
       

    def __len__(self):
        return self.size


class ValDataset(DatasetAll):
    def __init__(self, data_path, vocab, doc_max_timesteps, sent_max_len, ent_max_len, filter_word_path, w2s_path, s2s_path, thres_sim):

        super().__init__(data_path, vocab, doc_max_timesteps, sent_max_len, ent_max_len, filter_word_path, w2s_path, s2s_path, thres_sim)
                
        self.example_list = readJson(data_path)

    def get_example(self, index):
        e = self.example_list[index]
        e["summary"] = e.setdefault("summary", [])
        example = DatasetItem(e["text"],e["entity"], e["section_name"], e["summary"], self.vocab, self.sent_max_len, self.ent_max_len, e['label'])
        return example


class LoadHiExampleSet(torch.utils.data.Dataset):
    def __init__(self, data_root):
        super().__init__()
        self.data_root = data_root
        self.gfiles = [f for f in os.listdir(self.data_root) if f.endswith("graph.bin")]
        logger.info("[INFO] Start loading %s", self.data_root)

    def __getitem__(self, index):
        graph_file = os.path.join(self.data_root, "%d.graph.bin" % index)
        g, label_dict = load_graphs(graph_file)
        return g[0], index

    def __len__(self):
        return len(self.gfiles)




def readText(fname):
    data = []
    with open(fname, encoding="utf-8") as f:
        for line in f:
            data.append(line.strip())
    return data

def readJson(fname):
    data = []
#     with gzip.open(fname) as f:
#         for line in f:
#             # data.append(json.loads(line))
#             json_str = line.decode('utf-8')            # 2. string (i.e. JSON)
#             data.append(json.loads(json_str))
    with open(fname,'r', encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


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
