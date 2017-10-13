import codecs
import sys
from collections import defaultdict

class Corpus:
    """Simple Bitext corpus reader."""
    def __init__(self, source_file, target_file = None):
        self.source_sentences, self.source_sentences_maxlen = self.read_corpus(source_file)
        self.target_sentences, self.target_sentences_maxlen = self.read_corpus(target_file)
        self.source_word2idx, self.source_idx2word = self.get_dictionaries(self.source_sentences)
        self.target_word2idx, self.target_idx2word = self.get_dictionaries(self.target_sentences)

    def read_corpus(self, file_name):
        if file_name is None:
            return (None, 0)
        sentences = []
        maxlen = 0
        with codecs.open(file_name, "r", "utf-8") as fh:
            for line in fh:
                words = line.strip().split()
                maxlen = max(maxlen, len(words))
                sentences.append(words)
        return (sentences, maxlen)

    def write_corpus(self, file_name, sentences):
        with codecs.open(file_name + '.new', "w", "utf-8") as fh:
            for sentence in sentences:
                fh.write(' '.join(sentence))
                fh.write('\n')

    def get_dictionaries(self, sentences):
        if sentences is None:
            return None
        word2idx = defaultdict(int)
        idx = 0
        for sentence in sentences:
            for word in sentence:
                if word not in word2idx:
                    word2idx[word] = idx
                    idx += 1

        idx2word = {id: w for w, id in word2idx.iteritems()}
        return (word2idx, idx2word)
