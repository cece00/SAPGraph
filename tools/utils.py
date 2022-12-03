#!/usr/bin/python
# -*- coding: utf-8 -*-
import re
import os
import shutil
import copy
import datetime
import numpy as np

from rouge import Rouge

from .logger import *

import sys
sys.setrecursionlimit(10000)

_ROUGE_PATH = ""


REMAP = {"-lrb-": "(", "-rrb-": ")", "-lcb-": "{", "-rcb-": "}", 
        "-lsb-": "[", "-rsb-": "]", "``": '"', "''": '"'} 


def clean(x): 
    x = x.lower()
    return re.sub( 
            r"-lrb-|-rrb-|-lcb-|-rcb-|-lsb-|-rsb-|``|''", 
            lambda m: REMAP.get(m.group()), x)

def rouge_eval(hyps, refer):
    rouge = Rouge()
    try:
        score = rouge.get_scores(hyps, refer)[0]
        mean_score = np.mean([score["rouge-1"]["f"], score["rouge-2"]["f"], score["rouge-l"]["f"]])
    except:
        mean_score = 0.0
    return mean_score

def rouge_all(hyps, refer):
    rouge = Rouge()
    score = rouge.get_scores(hyps, refer)[0]
    return score

def eval_label(match_true, pred, true, total, match):
    match_true, pred, true, match = match_true.float(), pred.float(), true.float(), match.float()
    try:
        accu = match / total
        precision = match_true / pred
        recall = match_true / true
        F = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        accu, precision, recall, F = 0.0, 0.0, 0.0, 0.0
        logger.error("[Error] float division by zero")
    return accu, precision, recall, F


def cal_label(article, abstract):
    hyps_list = article

    refer = abstract
    scores = []
    for hyps in hyps_list:
        mean_score = rouge_eval(hyps, refer)
        scores.append(mean_score)

    selected = []
    selected.append(int(np.argmax(scores)))
    selected_sent_cnt = 1

    best_rouge = np.max(scores)
    while selected_sent_cnt < len(hyps_list):
        cur_max_rouge = 0.0
        cur_max_idx = -1
        for i in range(len(hyps_list)):
            if i not in selected:
                temp = copy.deepcopy(selected)
                temp.append(i)
                hyps = "\n".join([hyps_list[idx] for idx in np.sort(temp)])
                cur_rouge = rouge_eval(hyps, refer)
                if cur_rouge > cur_max_rouge:
                    cur_max_rouge = cur_rouge
                    cur_max_idx = i
        if cur_max_rouge != 0.0 and cur_max_rouge >= best_rouge:
            selected.append(cur_max_idx)
            selected_sent_cnt += 1
            best_rouge = cur_max_rouge
        else:
            break

    return selected
