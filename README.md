We have modified their code to use our data and a Bert-based encoder instead of embeddings and lstm.

Changes are in the adapted* files.



CHANGELOG:
  adapted_main.py
    * removed many hyperparams that no longer make sense, added other for bert, removed downsampling of data
    *

  adapted_reader.py
    * basically switched out with our partial_jsonl dataset reader



Run commands:
```
python adapted_main.py\
  --device "cuda:0"\
  --train-file ../data/conll2003/eng/entity.train.jsonl\
  --dev-file ../data/conll2003/eng/entity.dev.jsonl\
  --test-file ../data/conll2003/eng/entity.test.jsonl\
  --learning_rate "2e-5"\
  --l2 0.0\
  --num_epochs 10\
  --model_folder eng-c_debug\
  --bert-model roberta-base



## Better Modeling of Incomplete Annotation for Named Entity Recognition 

This repository implements an LSTM-CRF model for named entity recognition. The model is same as the one by [Lample et al., (2016)](http://www.anthology.aclweb.org/N/N16/N16-1030.pdf) except we do not have the last `tanh` layer after the BiLSTM.
The code provided is used for the paper "[Better Modeling of Incomplete Annotation for Named Entity Recognition](http://www.statnlp.org/research/ie/zhanming19naacl-ner.pdf)" published in 2019 Annual Conference of the North American Chapter of the Association for Computational Linguistics (*NAACL*).

__NOTE: To extend a more general use case, a PyTorch version is implemented in this repo. The previous implementation using DyNet can be found in first release [here](https://github.com/allanj/ner_incomplete_annotation/tree/aa20c015b3f373ac4a1893e629ac8f2dd137faab). Right now, I have implemented a the "hard" approach as in the paper. "Soft" approach would be coming soon.__

Our codebase is built based on the [pytorch LSTM-CRF](https://github.com/allanj/pytorch_lstmcrf) repo.


### Requirements
* PyTorch >= 1.1
* Python 3

Put your dataset under the data folder. You can obtain the `conll2003` and `conll2002` datasets from other sources. We have put our collected industry datasets `ecommerce` and `youku` under the data directory. 

Also, put your embedding file under the data directory to run. You need to specify the path for the embedding file.

### Running our approaches
```bash
python3 main.py --embedding_file ${PATH_TO_EMBEDDING} --dataset conll2003 --variant hard
```
Change `hard` to `soft` for our soft variant. 
(This version actually also supports using contextual representation. But I'm still testing during this weekend.)


### Future Work

- [x] add soft approach
- [ ] add other baselines.


### Citation
If you use this software for research, please cite our paper as follows:

The implementation in our paper is implemented with DyNet. Check out our previous [release](https://github.com/allanj/ner_incomplete_annotation/tree/aa20c015b3f373ac4a1893e629ac8f2dd137faab).
```
@inproceedings{jie2019better,
  title={Better Modeling of Incomplete Annotations for Named Entity Recognition},
  author={Jie, Zhanming and Xie, Pengjun and Lu, Wei and Ding, Ruixue and Li, Linlin},
  booktitle={Proceedings of NAACL},
  year={2019}
}
```
