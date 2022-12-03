import os
import json
import nltk
import random
import argparse


def catDoc(textlist): # textlist should be a paper with secitons 
    res = []
    for tlist in textlist:
        res.extend(tlist[1:]) # tlist[0] should be the section name
    return res

def getEnt(ent_list): # [[[start_id, end_id, ent0][start_id, end_id, ent1][...]],[sen2...]...]
    e_list = []
    for sec in ent_list:
        for sen in sec:
            #example of sen: [[0, 2, 'Mycoplasma pneumoniae'], [7, 8, 'upper'], [9, 10, 'lower'], [10, 13, 'respiratory tract infections']]
            for item in sen:
                e_list.extend(item[2].split())

    return e_list

def PrintInformation(keys, allcnt):
    # Vocab > 10
    cnt = 0
    first = 0.0
    for key, val in keys:
        if val >= 10:
            cnt += 1
            first += val
    print("appearance > 10 cnt: %d, percent: %f" % (cnt, first / allcnt))  # 416,303

    # first 30,000, last freq 31
    if len(keys) > 30000:
        first = 0.0
        for k, v in keys[:30000]:
            first += v
        print("First 30,000 percent: %f, last freq %d" % (first / allcnt, keys[30000][1]))

    # first 50,000, last freq 383
    if len(keys) > 50000:
        first = 0.0
        for k, v in keys[:50000]:
            first += v
        print("First 50,000 percent: %f, last freq %d" % (first / allcnt, keys[50000][1]))

    # first 100,000, last freq 107
    if len(keys) > 100000:
        first = 0.0
        for k, v in keys[:100000]:
            first += v
        print("First 100,000 percent: %f, last freq %d" % (first / allcnt, keys[100000][1]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default='data/cord-19/train.label.jsonl', help='File to deal with') # key: "text":2 d list, "summary":1d list; "label":1d list; "entity":3d list :[[[start_id, end_id, ent0][start_id, end_id, ent1][...]],[sen2...]...]
    parser.add_argument('--dataset', type=str, default='cord-19', help='dataset name')

    args = parser.parse_args()

    save_dir = os.path.join("cache", args.dataset)
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    saveFile = os.path.join(save_dir, "vocab")
    print("Save vocab of dataset %s to %s" % (args.dataset, saveFile))

    entFile = os.path.join(save_dir, "vocab_ent")
    print("Save entities vocab of dataset %s to %s" % (args.dataset, entFile))

    text = []
    summary = []
    entity = []
    allword = []
    cnt = 0
    with open(args.data_path, encoding='utf8') as f:
        for line in f:
            e = json.loads(line)
            
            if isinstance(e["text"], list) and isinstance(e["text"][0], list):
                sents = catDoc(e["text"])
                secs = catDoc(e['section_name'])
                sents.extend(secs)
            else:
                pass
            text = " ".join(sents)
            summary = " ".join(e["summary"])
            allword.extend(text.split())
            allword.extend(summary.split())
            
            entity.extend(getEnt(e['entity']))
            cnt += 1
            if cnt%2000 == 0:
                print(cnt)
    
    print("Training set of dataset has %d example" % cnt)
    fdist1 = nltk.FreqDist(allword)

    fout = open(saveFile, "w")
    keys = fdist1.most_common()
    for key, val in keys: # key is the word while value is the frequency (times)
        try:
            fout.write("%s\t%d\n" % (key, val))
        except UnicodeEncodeError as e:
            continue
    fout.close()
    
    
    # write entities into file, to make entities in vocab
    fout_ent = open(entFile, "w")
    fdist2 = nltk.FreqDist(entity)
    keys2 = fdist2.most_common()
    k_left = 0
    for key, val in keys2: # key is the word while value is the frequency (times)
        try:
            pass_sig = False
            filt = [',', ':', ';', '?', '&', '!', '*', '@', '$', '%', '\\', '`', '``', '|', '/']
            for item in filt:
                if item in repr(key):
                    pass_sig = True
                    break
            if not pass_sig:
                fout_ent.write("%s\t%s\t%d\n" % (key, key.lower(), val))
                k_left +=1
        except UnicodeEncodeError as e:
            continue
    fout_ent.close()

    
    '''
    allcnt = fdist1.N() # 788,159,121
    allset = fdist1.B() # 5,153,669
    '''
    
