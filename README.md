# SAPGraph
This repository contains code and data for paper [SAPGraph: Structure-aware Extractive Summarization for Scientific Papers with Heterogeneous Graph](https://aclanthology.org/2022.aacl-main.44/). 

## Environment
For pyTorch and DGL, I use
```
torch==1.9.0+cu111
dgl==1.1.2+cu111
```
Should install the right dgl package for your system and cuda, through https://www.dgl.ai/pages/start.html. Note that dgl-0.x.x is a bit differet with dgl-1.x.x.

Run 
```
pip install -r requirements.txt
```

## Data Prepare
Raw data is avaliable on Google Drive:
https://drive.google.com/file/d/1_5gZkOFYeB4FKrt-60J7bxC09iULAniI/view?usp=sharing
This includes the source document (parsed from paper pdf) and its summary as a pair.

After putting the data into dir *data/cordSum*, you need to run *PrepareDataset.sh* to prepare graph meta data for each sample, including node and edge construction. And the result will save into dir *cache/cordSum*.
```
bash PrepareDataset.sh cordSum cord/cordSum
```

## Training

Some code are borrowed from [HSG](https://github.com/dqwang122/HeterSumGraph). Thanks for their work.
```
bash train_cord.sh
```
Please make sure the settings is what you want before running.

## Testing

```
bash test_cord.sh
```
