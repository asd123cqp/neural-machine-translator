# Spring 2017 NLP HW4
Name: Qipeng Chen
UNI: qc2201

Note:

- All the trained models were moved to `output` directory.
- This doc is written in markdown.

Best score: 30.7737948678 with beam search and attention, 256d/256d, beam size 5.

Usage:

​	In `src` directory,

- `./run.sh` to train a model for part 1
- `./run.sh att` to train a model for part 2
- `./run.sh beam` to load a trained part 2 model and do beam search. Modify the global variables in `nmt_beam.py` if you want to change beam size, model path or dev data path.

## Part 1

### First attempt

At first I used the default configuration, where `word_d = 50`, `gru_d = 50`, but changed `gru_layers = 1`. I used the seed number 1234567890 in this part for consistency.

#### Early termination

Since it's not the final model, for time saving, I add these lines so that the program can exit as soon as the score reach 22.

```python
if bleu_score > 22:
    break
```

For this version, its `bleu_score` after first epoch was 13.4586706948 and it was able reach a `bleu_score` of 22+ at epoch 12.

### Extra credit: pretrained embeddings

#### English embeddings

Here's the embeddings I used for english: [Glove](http://nlp.stanford.edu/data/glove.6B.zip).

First, I converted the `.txt` file to a `word2vec` object like using the command `python -m gensim.scripts.glove2word2vec -i glove.6B.50d.txt -o glove.6B.50d.vec`.

I used the 50d version and loaded into our model like this (line 38~50 in `nmt_dynet.py`):

```python
tgt_emb = gensim.models.KeyedVectors.load_word2vec_format('/Users/C/Desktop/glove.6B.50d.vec')
tmp = []
for i in range(tgt_vocab_size):
    if tgt_idx2word[i] in tgt_emb:
        tmp.append(tgt_emb[tgt_idx2word[i]])
    else:
        tmp.append(self.target_embeddings[i].npvalue())
tmp = np.array(tmp)
np.save(open('../data/en.emb', 'wb'), tmp)
print "Target Embeddings saved"

tmp = np.load('../data/en.emb')
self.target_embeddings.init_from_array(tmp)
```

The embeddings array was saved to `data/en.emb`. With the pretrained only english embeddings and the same seed number, the `bleu_score` was improved quite a bit:  14.7326023517 after first epoch and reach 22 at the 7-th epoch. Amazing!

#### Chinese embeddings

The pretrained embeddings for chinese is harder to find. This is the one I end up using: [fastText](https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.zh.vec). However, it's 300d so I need to do dimension reduction before using it. Here's how I load it into the model (line 52~69 in `nmt_dynet.py`):

```python
from sklearn.decomposition import PCA

src_emb = gensim.models.KeyedVectors.load_word2vec_format('/Users/C/Desktop/wiki.zh.vec')
count = 0
tmp = []
pca = PCA(n_components=word_d)
for i in range(src_vocab_size):
    if src_idx2word[i] in src_emb:
        lowd = pca.fit_transform(src_emb[src_idx2word[i]])
        tmp.append(lowd)
        count += 1
    else:
        tmp.append(self.source_embeddings[i].npvalue())

print count, src_vocab_size
np.save(open('../data/cn.emb', 'wb'), tmp)
print "Source Embeddings saved"

tmp = np.load('../data/cn.emb')
self.source_embeddings.init_from_array(tmp)
```

Here I used PCA to reduce it's dimension to 50. The embeddings array was saved to `data/cn.emb`.

Sadly, the chinese embeddings are not as good. With only the pretrained chinese embeddings and the same seed number, the `bleu_score` actually got worse: 12.8120952112 after first epoch and didn't reach 22 after 12 epochs and was then killed by me.

Some possible cause:

1. Too many information was lost during dimension reduction;
2. Many of the words (~20%) in the training set are not in the pretrained data.

#### Conclusion

Here's the result for some experiments I did:

| Embeddings       | First epoch   | Epoch to reach 22 | Best in 20 epochs  |
| ---------------- | ------------- | ----------------- | ------------------ |
| No pre-train emb | 13.4586706948 | 12                | 23.3082395632 (16) |
| Only English emb | 14.7326023517 | 7                 | 23.2330023783 (13) |
| Only Chinese emb | 12.8120952112 | NA - killed at 12 | NA                 |
| Both             | 13.8892310285 | 11                | 23.2175349023 (17) |

As a result, although the model with english pre-train embeddings perform the best at first, all models converge to very a close accuracy. Therefore, in this case, the pretrained embeddings help us train the model faster, but given enough time, the final results are not that big of a difference.

-----

## Part 2

Really not much to say about part 2. My code should work exactly as specified.

-----

## Extra credit: Beam search

My implementation was saved as `src/nmt_beam.py`. It will only load the model trained in part 2 and do translation, since it makes no sense to train again. The original results will also be printed out for comparison. The default beam size is 5.

Usage:

- `./run.sh beam` to load a trained part 2 model and do beam search.

- Change `DEV_SRC` and `DEV_TGT` at line 14/15 to change test data path.

- Change `BSIZE` at line 18 to change beam size;

-----

## Final models

Adding word/hidden dimensions could increase the power of a model. Here I used 100d/100d for the model in part 1(no attention) and the best `bleu_score` is 26.4606497698 without beam search, way above the required 22. The model was saved to `output/final_nmt_dynet`.

I then trained a model with 256d/256d for the model in part 2 (with attention), and the result is amazing: `bleu_score` is 30.7737948678 with beam search (beam size 5) and 29.7606096855 without. The model was saved to `output/final_nmt_dynet_attention`.
