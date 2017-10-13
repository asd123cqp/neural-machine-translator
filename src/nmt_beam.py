from dynet import *
import argparse
from utils import Corpus
import random
import numpy as np
from bleu import get_bleu_score
import json
import pickle
import nmt_dynet_attention as nmt
# import nmt_dynet as nmt_noatt
from heapq import *

# Path to dev data
DEV_SRC = '../data/dev.src'
DEV_TGT = '../data/dev.tgt'

# Beam size
BSIZE = 5

# Other constants
MODEL = "../output/final_nmt_dynet_attention"
TRAIN_SRC = '../data/train.src'
TRAIN_TGT = '../data/train.tgt'

def load_model(train_set, dev_set, model):
    encoder_decoder = nmt.nmt_dynet_attention(len(train_set.source_word2idx), len(train_set.target_word2idx), train_set.source_word2idx, train_set.source_idx2word, train_set.target_word2idx, train_set.target_idx2word, nmt.W_DIM, nmt.H_DIM, 1)
    encoder_decoder.model.load(model)

    return encoder_decoder

def greedy_gen(model, src_sentence):

    renew_cg()
    src = model.encode(src_sentence)
    dec = model.dec_RNN.initial_state()
    W, b = parameter(model.output_w), parameter(model.output_b)
    dec = dec.add_input(concatenate([vecInput(model.word_d), src[-1]]))

    predict = [model.tgt_word2idx["<s>"]]
    for _ in range(10 * len(src_sentence)):
        ci = model.attend(src, dec.output())
        x = concatenate([model.target_embeddings[predict[-1]], ci])
        dec = dec.add_input(x)
        prob = softmax(W * dec.output() + b).npvalue()
        i = np.argmax(prob)
        if i == model.STOP:
            break
        predict.append(i)

    return [model.tgt_idx2word[i] for i in predict] + ["</s>"]

def beam_search(model, src_sentence, beam_size):
    renew_cg()
    src = model.encode(src_sentence)
    dec = model.dec_RNN.initial_state()
    W, b = parameter(model.output_w), parameter(model.output_b)
    dec = dec.add_input(concatenate([vecInput(model.word_d), src[-1]]))

    q, done = [(0, [model.tgt_word2idx["<s>"]], dec)], []
    for _ in range(10 * len(src_sentence)):
        nq = []
        while q:
            cur = heappop(q)
            cur_dec = cur[2]
            ci = model.attend(src, cur_dec.output())
            x = concatenate([model.target_embeddings[cur[1][-1]], ci])
            cur_dec = cur_dec.add_input(x)
            prob = log_softmax(W * cur_dec.output() + b).npvalue()
            for i, p in enumerate(prob):
                heappush(nq, (p + cur[0], cur[1] + [i], cur_dec))
                if len(nq) > beam_size - len(done):
                    heappop(nq)
        for hypo in nq:
            if hypo[1][-1] == model.STOP:
                heappush(done, (hypo[0], hypo[1][:-1], hypo[2]))
            else:
                heappush(q, hypo)

        if len(done) == beam_size:
        # if len(done) > 0:
            break

    if not done:
        best = nlargest(1, q)[0][1]
    else:
        best = nlargest(1, done)[0][1]

    return [model.tgt_idx2word[i] for i in best] + ["</s>"]

def gen_all(model, src_sentences, beam_size):
    greedy, beam = [], []
    count = 0
    print "Beam size:", beam_size
    print "Translating", len(src_sentences), "sentences . . ."
    for s in src_sentences:
        beam.append(beam_search(model, s, beam_size))
        greedy.append(greedy_gen(model, s))
        count += 1
        if count % 10 == 0:
            print "Sentences processed:", count

    return greedy, beam

# def greedy_gen_noatt(model, src_sentence):

#     renew_cg()
#     src = model.encode(src_sentence)[-1]
#     dec = model.dec_RNN.initial_state()
#     W, b = parameter(model.output_w), parameter(model.output_b)
#     dec = dec.add_input(concatenate([vecInput(model.word_d), src]))

#     predict = [model.tgt_word2idx["<s>"]]

#     for _ in range(10 * len(src_sentence)):
#         x = concatenate([model.target_embeddings[predict[-1]], src])
#         dec = dec.add_input(x)
#         prob = softmax(W * dec.output() + b).npvalue()
#         i = np.argmax(prob)
#         if i == model.STOP:
#             break
#         predict.append(i)

#     return [model.tgt_idx2word[i] for i in predict] + ["</s>"]

def main():

    # load trained model
    print 'Loading neural machine translator with attention:'
    train_set = Corpus(TRAIN_SRC, TRAIN_TGT)
    dev_set = Corpus(DEV_SRC, DEV_TGT)
    m = load_model(train_set, dev_set, MODEL)
    print 'Model loaded!'

    # dev_set.target_sentences = dev_set.target_sentences[100:200]
    # dev_set.source_sentences = dev_set.source_sentences[100:200]

    # translate sentence
    print '\nTranslating . . .\n'
    sample_output = np.random.choice(len(dev_set.target_sentences), 5, False)
    greedy, beam = gen_all(m, dev_set.source_sentences, BSIZE)
    for sample in sample_output:
        print 'Target: {}'.format(' '.join(dev_set.target_sentences[sample]))
        print 'Greedy: {}'.format(' '.join(greedy[sample]))
        print 'Beam search: {}'.format(' '.join(beam[sample]))
        print '----------'

    greedy_score = get_bleu_score(greedy, dev_set.target_sentences)
    beam_score = get_bleu_score(beam, dev_set.target_sentences)


    print 'Greedy bleu score: ', greedy_score
    print 'Beam search bleu score: ', beam_score


    # test for greedy search without attention
    # m_noatt = nmt_noatt.nmt_dynet(len(train_set.source_word2idx), len(train_set.target_word2idx), train_set.source_word2idx, train_set.source_idx2word, train_set.target_word2idx, train_set.target_idx2word, nmt_noatt.W_DIM, nmt_noatt.H_DIM, 1)
    # m_noatt.model.load("../output/final_nmt_dynet")
    # noatt_sent = [greedy_gen_noatt(m_noatt, s) for s in dev_set.source_sentences]
    # noatt_score = get_bleu_score(noatt_sent, dev_set.target_sentences)
    # print 'Greedy without attention: ', noatt_score


if __name__ == '__main__':
    main()
