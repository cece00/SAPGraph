import os
import argparse
import json
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from scipy.linalg import norm
import multiprocessing
import traceback

from tqdm import tqdm


def similarity_mat(doc_len, doc_vec):
    """
    :param doc_len:list, the length number of doc_vec
    :param doc_vec:list[list], word-level text, sum(doc_len) == len(doc_vec)
    :return: 
        tfidf: array [sent_number, max_word_number]
    """
    def cos_similarity(v1, v2):
        s =  norm(v1) * norm(v2)
        if s != 0:
            return np.dot(v1, v2) /s
        else:
            return 0 

    count, off_set = 0,0
    tfidf = {}
    while count < len(doc_len):
        m = doc_len[count]
        sim_mat = [[0 for _ in range(m)] for i in range(m)]
        for i in range(m):
            sim_mat[i][i] = 1.0
            for j in range(i+1, m):
                sim_mat[i][j] = round(cos_similarity(doc_vec[i+off_set],doc_vec[j+off_set]), 3)
        tfidf[count] = sim_mat
        off_set = sum(doc_len[:count+1])
        count += 1
    return tfidf

def GetType(path):
    filename = path.split("/")[-1]
    return filename.split(".")[0]

def catDoc(textlist):
    res = []
    t_len = []
    for tlist in textlist:
        res.extend(tlist)
        t_len.append(len(tlist))
    return res, t_len

def get_tfidf_embedding(text):
    """
    :param text: list, sent_number * word
    :return: 
        vectorizer: 
            vocabulary_: word2id
            get_feature_names(): id2word
        tfidf: array [sent_number, max_word_number]
    """
    vectorizer = CountVectorizer(lowercase=True)
    word_count = vectorizer.fit_transform(text)
    tfidf_transformer = TfidfTransformer()
    tfidf = tfidf_transformer.fit_transform(word_count)
    tfidf_weight = tfidf.toarray()
    return vectorizer, tfidf_weight
    

def compress_array(doc_len, td, id2word, ents):
    """
    :param doc: list of length of each section 
    :param td: matrix, [N, M], N is document number, M is word number
    :param id2word: word id to word
    :return: 
    """
    d = {}
    count,i = 0,0
    res = {}

    for i in range(len(td)):
        d[i] = {}
        for j in range(len(td[i])):
            if td[i][j] != 0:
                if id2word[j] in ents:
                    d[i][id2word[j]] = td[i][j]
    return d


def main(args):
    save_dir = os.path.join("cache", args.dataset)
    entFile = os.path.join(save_dir, "vocab_ent")
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    fw2s_name = GetType(args.data_path) + ".w2s.tfidf.jsonl"
    fw2s_savename = os.path.join(save_dir, fw2s_name)

    fs2s_name = GetType(args.data_path) + ".s2s.tfidf.jsonl"
    fs2s_savename = os.path.join(save_dir, fs2s_name)

    # print("Save word2sent features of dataset %s to %s" % (args.dataset, saveFile))
    print("Save entity2sent features of dataset %s to %s" % (args.dataset, fw2s_savename))
    print("Save sent2sent features of dataset %s to %s" % (args.dataset, fs2s_savename))

    f_ent = open(entFile, "r")
    ents = []
    for line in f_ent:
        # print(line, type(line))
        # print(line.split('\t'))
        c = int(line.strip().split('\t')[2])
        ents.append(line.split('\t')[1])
        if c<300:
            break
    print('read entity over')

    fw2s_out = open(fw2s_savename, "w")
    fs2s_out = open(fs2s_savename, "w")
    print_every = 5000
    count = 0
    with open(args.data_path) as f:
        for line in tqdm(f):
            if count%print_every == 0:
                print('now is constucting edge for line ',count)
            e = json.loads(line)
            if isinstance(e["text"], list) and isinstance(e["text"][0], list):
                sents, doc_len = catDoc(e["text"])

            cntvector, tfidf_weight = get_tfidf_embedding(sents)

            id2word = {}
            for w, tfidf_id in cntvector.vocabulary_.items():   # word -> tfidf matrix row number
                id2word[tfidf_id] = w  # tfidf_id is not word_id in vocab
            tfidfvector = compress_array(doc_len, tfidf_weight, id2word, ents) # 2d dict
            fw2s_out.write(json.dumps(tfidfvector) + "\n")

            s2s_mat = similarity_mat(doc_len, tfidf_weight) # dict with sec_id as key
            fs2s_out.write(json.dumps(s2s_mat) + "\n")

            count += 1
            

def main_train(args):
    process_list = []
    for i in range(5):
        process_list.append(multiprocessing.Process(target=train_worker, args=(i,args,)))

    for p in process_list:
        p.start()

def train_worker(worker_id,args):
    save_dir = os.path.join("cache", args.dataset)
    entFile = os.path.join(save_dir, "vocab_ent")
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    fw2s_name = GetType(args.data_path) + ".w2s.tfidf.jsonl"+str(worker_id)
    fw2s_savename = os.path.join(save_dir, fw2s_name)

    fs2s_name = GetType(args.data_path) + ".s2s.tfidf.jsonl"+str(worker_id)
    fs2s_savename = os.path.join(save_dir, fs2s_name)

    print("Save entity2sent features of dataset %s to %s" % (args.dataset, fw2s_savename))
    print("Save sent2sent features of dataset %s to %s" % (args.dataset, fs2s_savename))

    f_ent = open(entFile, "r")
    ents = []
    for line in f_ent:
        c = int(line.strip().split('\t')[2])
        ents.append(line.split('\t')[1])
        if c<300:
            break
    print('read entity over')

    fw2s_out = open(fw2s_savename, "w")
    fs2s_out = open(fs2s_savename, "w")
    print_every = 5000
    count = 0

    start = worker_id*20000
    if worker_id!=4:
        end = start+20000
    else:
        end = 85908

    with open(args.data_path) as f:
        for line in tqdm(f.readlines()[start:end]):
            if count%print_every == 0:
                print('now is constucting edge for line ',count)
            e = json.loads(line)
            if isinstance(e["text"], list) and isinstance(e["text"][0], list):
                sents, doc_len = catDoc(e["text"])

            cntvector, tfidf_weight = get_tfidf_embedding(sents)

            id2word = {}
            for w, tfidf_id in cntvector.vocabulary_.items():   # word -> tfidf matrix row number
                id2word[tfidf_id] = w  # tfidf_id is not word_id in vocab
            tfidfvector = compress_array(doc_len, tfidf_weight, id2word, ents) # 2d dict
            fw2s_out.write(json.dumps(tfidfvector) + "\n")

            s2s_mat = similarity_mat(doc_len, tfidf_weight) # dict with sec_id as key
            fs2s_out.write(json.dumps(s2s_mat) + "\n")

            count += 1


def train_merge(args):
    save_dir = os.path.join("cache", args.dataset)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/cordSum/train.label.jsonl', help='File to deal with')
    parser.add_argument('--dataset', type=str, default='cordSum', help='dataset name')
    args = parser.parse_args()

    if 'train' in args.data_path:
        main_train(args)
    else:
        main(args)
    print('caledge is over!')
        
