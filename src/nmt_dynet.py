#!/anaconda/bin/python
# -*- coding: UTF-8 -*-
from dynet import *
import argparse
from utils import Corpus
import random
import numpy as np
from bleu import get_bleu_score
import json
import gensim
# from sklearn.decomposition import PCA

RNN_BUILDER = GRUBuilder
W_DIM = 100
H_DIM = 100

class nmt_dynet:

    def __init__(self, src_vocab_size, tgt_vocab_size, src_word2idx, src_idx2word, tgt_word2idx, tgt_idx2word, word_d, gru_d, gru_layers):

        # initialize variables
        self.gru_layers = gru_layers
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.src_word2idx = src_word2idx
        self.src_idx2word = src_idx2word
        self.tgt_word2idx = tgt_word2idx
        self.tgt_idx2word = tgt_idx2word
        self.word_d = word_d
        self.gru_d = gru_d

        self.model = Model()

        # the embedding paramaters
        self.source_embeddings = self.model.add_lookup_parameters((self.src_vocab_size, self.word_d))
        self.target_embeddings = self.model.add_lookup_parameters((self.tgt_vocab_size, self.word_d))

        # tgt_emb = gensim.models.KeyedVectors.load_word2vec_format('/Users/C/Desktop/glove.6B.50d.vec')
        # tmp = []
        # for i in range(tgt_vocab_size):
        #     if tgt_idx2word[i] in tgt_emb:
        #         tmp.append(tgt_emb[tgt_idx2word[i]])
        #     else:
        #         tmp.append(self.target_embeddings[i].npvalue())
        # tmp = np.array(tmp)
        # np.save(open('../data/en.emb', 'wb'), tmp)
        # print "Target Embeddings saved"

        # tmp = np.load('../data/en.emb')
        # self.target_embeddings.init_from_array(tmp)

        # src_emb = gensim.models.KeyedVectors.load_word2vec_format('/Users/C/Desktop/wiki.zh.vec')
        # count = 0
        # tmp = []
        # pca = PCA(n_components=word_d)
        # for i in range(src_vocab_size):
        #     if src_idx2word[i] in src_emb:
        #         lowd = pca.fit_transform(src_emb[src_idx2word[i]])
        #         tmp.append(lowd)
        #         count += 1
        #     else:
        #         tmp.append(self.source_embeddings[i].npvalue())

        # print count, src_vocab_size
        # np.save(open('../data/cn.emb', 'wb'), tmp)
        # print "Source Embeddings saved"

        # tmp = np.load('../data/cn.emb')
        # self.source_embeddings.init_from_array(tmp)

        # project the decoder output to a vector of tgt_vocab_size length
        self.output_w = self.model.add_parameters((tgt_vocab_size, word_d))
        self.output_b = self.model.add_parameters((tgt_vocab_size))

        # encoder network
        # the foreword rnn
        self.fwd_RNN = RNN_BUILDER(gru_layers, word_d, gru_d, self.model)
        # the backword rnn
        self.bwd_RNN = RNN_BUILDER(gru_layers, word_d, gru_d, self.model)

        # decoder network
        self.dec_RNN = RNN_BUILDER(gru_layers, word_d + gru_d * 2,
                                   gru_d, self.model)

        # others
        self.STOP = self.tgt_word2idx["</s>"]


    def encode(self, src_sentence):
        '''
        src_sentence: list of words in the source sentence (i.e output of .strip().split(' '))
        return encoding of the source sentence
        '''
        fwd, bwd = self.fwd_RNN.initial_state(), self.bwd_RNN.initial_state()
        idx = [self.src_word2idx[w] for w in src_sentence]
        emb = [self.source_embeddings[i] for i in idx]

        fs, bs = fwd.add_inputs(emb), bwd.add_inputs(emb[::-1])[::-1]
        states = [concatenate([f.output(), b.output()]) \
                  for f, b in zip(fs, bs)]

        return states


    def get_loss(self, src_sentence, tgt_sentence):
        '''
        src_sentence: words in src sentence
        tgt_sentence: words in tgt sentence
        return loss for this source target sentence pair
        '''

        renew_cg()
        src = self.encode(src_sentence)[-1]
        dec = self.dec_RNN.initial_state()
        W, b = parameter(self.output_w), parameter(self.output_b)
        idx = [self.tgt_word2idx[w] for w in tgt_sentence]
        emb = [self.target_embeddings[i] for i in idx]
        x = [concatenate([vecInput(self.word_d), src])] + \
            [concatenate([w, src]) for w in emb[:-1]]
        dec = dec.add_inputs(x)
        prob = [softmax(W * d.output() + b) for d in dec[1:]]
        loss = [-log(pick(p, self.tgt_word2idx[tgt_sentence[i+1]])) \
                for i, p in enumerate(prob)]

        return esum(loss)


    def generate(self, src_sentence):
        '''
        src_sentence: list of words in the source sentence (i.e output of .strip().split(' '))
        return list of words in the target sentence
        '''

        renew_cg()

        # YOUR IMPLEMENTATION GOES HERE
        src = self.encode(src_sentence)[-1]
        dec = self.dec_RNN.initial_state()
        W, b = parameter(self.output_w), parameter(self.output_b)
        dec = dec.add_input(concatenate([vecInput(self.word_d), src]))

        predict = [self.tgt_word2idx["<s>"]]

        for _ in range(10 * len(src_sentence)):
            x = concatenate([self.target_embeddings[predict[-1]], src])
            dec = dec.add_input(x)
            prob = softmax(W * dec.output() + b).npvalue()
            i = np.argmax(prob)
            if i == self.STOP:
                break
            predict.append(i)

        return [self.tgt_idx2word[i] for i in predict] + ["</s>"]


    def translate_all(self, src_sentences):
        translated_sentences = []
        for src_sentence in src_sentences:
            # print src_sentence
            translated_sentences.append(self.generate(src_sentence))

        return translated_sentences

    # save the model, and optionally the word embeddings
    def save(self, filename):

        self.model.save(filename)
        embs = {}
        if self.src_idx2word:
            src_embs = {}
            for i in range(self.src_vocab_size):
                src_embs[self.src_idx2word[i]] = self.source_embeddings[i].value()
            embs['src'] = src_embs

        if self.tgt_idx2word:
            tgt_embs = {}
            for i in range(self.tgt_vocab_size):
                tgt_embs[self.tgt_idx2word[i]] = self.target_embeddings[i].value()
            embs['tgt'] = tgt_embs

        if len(embs):
            with open(filename + '_embeddings.json', 'w') as f:
                json.dump(embs, f)

def get_val_set_loss(network, val_set):
        loss = []
        for src_sentence, tgt_sentence in zip(val_set.source_sentences, val_set.target_sentences):
            loss.append(network.get_loss(src_sentence, tgt_sentence).value())

        return sum(loss)

def main(train_src_file, train_tgt_file, dev_src_file, dev_tgt_file, model_file, num_epochs, embeddings_init = None, seed = 0):
    print('reading train corpus ...')
    train_set = Corpus(train_src_file, train_tgt_file)
    # assert()
    print('reading dev corpus ...')
    dev_set = Corpus(dev_src_file, dev_tgt_file)

    print 'Initializing simple neural machine translator:'
    # src_vocab_size, tgt_vocab_size, tgt_idx2word, word_d, gru_d, gru_layers
    encoder_decoder = nmt_dynet(len(train_set.source_word2idx), len(train_set.target_word2idx), train_set.source_word2idx, train_set.source_idx2word, train_set.target_word2idx, train_set.target_idx2word, W_DIM, H_DIM, 1)

    trainer = SimpleSGDTrainer(encoder_decoder.model)

    sample_output = np.random.choice(len(dev_set.target_sentences), 5, False)
    losses = []
    best_bleu_score = 0
    for epoch in range(num_epochs):
        print 'Starting epoch', epoch
        # shuffle the training data
        combined = list(zip(train_set.source_sentences, train_set.target_sentences))
        random.shuffle(combined)
        train_set.source_sentences[:], train_set.target_sentences[:] = zip(*combined)

        print 'Training . . .'
        sentences_processed = 0
        for src_sentence, tgt_sentence in zip(train_set.source_sentences, train_set.target_sentences):
            loss = encoder_decoder.get_loss(src_sentence, tgt_sentence)
            loss_value = loss.value()
            loss.backward()
            trainer.update()
            sentences_processed += 1
            if sentences_processed % 4000 == 0:
                print 'sentences processed: ', sentences_processed

        # Accumulate average losses over training to plot
        val_loss = get_val_set_loss(encoder_decoder, dev_set)
        print 'Validation loss this epoch', val_loss
        losses.append(val_loss)

        print 'Translating . . .'
        translated_sentences = encoder_decoder.translate_all(dev_set.source_sentences)

        print('translating {} source sentences...'.format(len(sample_output)))
        for sample in sample_output:
            print('Target: {}\nTranslation: {}\n'.format(' '.join(dev_set.target_sentences[sample]), ' '.join(translated_sentences[sample])))

        bleu_score = get_bleu_score(translated_sentences, dev_set.target_sentences)
        print 'bleu score: ', bleu_score
        if bleu_score > best_bleu_score:
            best_bleu_score = bleu_score
            # save the model
            encoder_decoder.save(model_file)

        # if bleu_score > 22:
        #     break

    print 'best bleu score: ', best_bleu_score

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = '')
#     parser.add_argument('model_type')
    parser.add_argument('train_src_file')
    parser.add_argument('train_tgt_file')
    parser.add_argument('dev_src_file')
    parser.add_argument('dev_tgt_file')
    parser.add_argument('model_file')
    parser.add_argument('--num_epochs', default = 20, type = int)
    parser.add_argument('--embeddings_init')
    parser.add_argument('--seed', default = 0, type = int)
    parser.add_argument('--dynet-mem')

    args = vars(parser.parse_args())
    args.pop('dynet_mem')

    main(**args)
