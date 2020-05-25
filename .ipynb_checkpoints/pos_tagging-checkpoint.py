import matplotlib.pyplot as plt
import numpy as np

from IPython.core.display import HTML
from itertools import chain
from collections import Counter, defaultdict
from pomegranate import State, HiddenMarkovModel, DiscreteDistribution

import os
import matplotlib.pyplot as plt
import matplotlib.image as mplimg
import networkx as nx
import random

from io import BytesIO
from itertools import chain
from collections import namedtuple, OrderedDict

Sentence = namedtuple("Sentence", "words tags")

def read_data(filename):
    with open(filename, 'r') as f:
        sentence_lines = [l.split("\n") for l in f.read().split("\n\n")]
    return OrderedDict(((s[0], Sentence(*zip(*[l.strip().split("\t")
                        for l in s[1:]]))) for s in sentence_lines if s[0]))


def read_tags(filename):
    with open(filename, 'r') as f:
        tags = f.read().split("\n")
    return frozenset(tags)

class Subset(namedtuple("BaseSet", "sentences keys vocab X tagset Y N stream")):
    def __new__(cls, sentences, keys):
        word_sequences = tuple([sentences[k].words for k in keys])
        tag_sequences = tuple([sentences[k].tags for k in keys])
        wordset = frozenset(chain(*word_sequences))
        tagset = frozenset(chain(*tag_sequences))
        N = sum(1 for _ in chain(*(sentences[k].words for k in keys)))
        stream = tuple(zip(chain(*word_sequences), chain(*tag_sequences)))
        return super().__new__(cls, {k: sentences[k] for k in keys}, keys, wordset, word_sequences,
                               tagset, tag_sequences, N, stream.__iter__)

    def __len__(self):
        return len(self.sentences)

    def __iter__(self):
        return iter(self.sentences.items())


class Dataset(namedtuple("_Dataset", "sentences keys vocab X tagset Y training_set testing_set N stream")):
    def __new__(cls, tagfile, datafile, train_test_split=0.8, seed=112890):
        tagset = read_tags(tagfile)
        sentences = read_data(datafile)
        keys = tuple(sentences.keys())
        wordset = frozenset(chain(*[s.words for s in sentences.values()]))
        word_sequences = tuple([sentences[k].words for k in keys])
        tag_sequences = tuple([sentences[k].tags for k in keys])
        N = sum(1 for _ in chain(*(s.words for s in sentences.values())))
        
        # split data into train/test sets
        _keys = list(keys)
        if seed is not None: random.seed(seed)
        random.shuffle(_keys)
        split = int(train_test_split * len(_keys))
        training_data = Subset(sentences, _keys[:split])
        testing_data = Subset(sentences, _keys[split:])
        stream = tuple(zip(chain(*word_sequences), chain(*tag_sequences)))
        return super().__new__(cls, dict(sentences), keys, wordset, word_sequences, tagset,
                               tag_sequences, training_data, testing_data, N, stream.__iter__)

    def __len__(self):
        return len(self.sentences)

    def __iter__(self):
        return iter(self.sentences.items())
    
# Step 1: Read and preprocess the dataset
data = Dataset("tags-universal.txt", "brown-universal.txt", train_test_split=0.8)

# Step 2: Build a Most Frequent Class tagger
def pair_counts(sequences_A, sequences_B): 
    pair_count_dict = defaultdict(Counter)
    
    for i in range(len(sequences_A)):
        for key, value in zip(sequences_A[i],sequences_B[i]):
            pair_count_dict[key][value] += 1
            
    return pair_count_dict

FakeState = namedtuple("FakeState", "name")

class MFCTagger:
    missing = FakeState(name="<MISSING>")
    
    def __init__(self, table):
        self.table = defaultdict(lambda: MFCTagger.missing)
        self.table.update({word: FakeState(name=tag) for word, tag in table.items()})
        
    def viterbi(self, seq):
        return 0., list(enumerate(["<start>"] + [self.table[w] for w in seq] + ["<end>"]))

word_counts = pair_counts(data.training_set.X, data.training_set.Y)

mfc_table = {}
for key, value in word_counts.items():
    mfc_table[key] = max(word_counts[key], key=word_counts[key].get)

mfc_model = MFCTagger(mfc_table)

def replace_unknown(sequence):
    return [w if w in data.training_set.vocab else 'nan' for w in sequence]

def simplify_decoding(X, model):
    _, state_path = model.viterbi(replace_unknown(X))
    return [state[1].name for state in state_path[1:-1]]

def accuracy(X, Y, model):
    correct = total_predictions = 0
    for observations, actual_tags in zip(X, Y):

        try:
            most_likely_tags = simplify_decoding(observations, model)
            correct += sum(p == t for p, t in zip(most_likely_tags, actual_tags))
        except:
            pass
        total_predictions += len(observations)
    return correct / total_predictions

# mfc_training_acc = accuracy(data.training_set.X, data.training_set.Y, mfc_model)

# mfc_testing_acc = accuracy(data.testing_set.X, data.testing_set.Y, mfc_model)

# Step 3: Build an HMM tagger
def unigram_counts(sequences):
    all_tags = []
    for sequence in sequences:
        all_tags.extend(sequence)
        
    tag_counts = {}
    unique_tags = list(set(all_tags))
    for tag in unique_tags:
        tag_counts[tag] = all_tags.count(tag)
    
    return tag_counts
        
tag_unigrams = unigram_counts(data.training_set.Y)

def bigram_counts(sequences):
    tag_counts = {}
    for sequence in sequences:
        for i in range(0,len(sequence)-1):
            t1 = sequence[i]
            t2 = sequence[i+1]
            tup = (t1,t2)
            tag_counts[tup] = tag_counts.get(tup,1) + 1
            
    return tag_counts

tag_bigrams = bigram_counts(data.training_set.Y)

def starting_counts(sequences):
    tag_counts = {}
    for sequence in sequences:
        tag = sequence[0]
        tag_counts[tag] = tag_counts.get(tag,1) + 1
        
    return tag_counts

tag_starts = starting_counts(data.training_set.Y)

def ending_counts(sequences):
    tag_counts = {}
    for sequence in sequences:
        tag = sequence[-1]
        tag_counts[tag] = tag_counts.get(tag,1) + 1
        
    return tag_counts

tag_ends = ending_counts(data.training_set.Y)

basic_model = HiddenMarkovModel(name="base-hmm-tagger")

prob_emission = {}
states = {}
for tag, word_counts in emission_counts.items():
    prob_emission = {word : word_count/sum(word_counts.values()) for word, word_count in word_counts.items()}
    states[tag] = State(DiscreteDistribution(prob_emission), name=tag)

unique_tags = list(data.training_set.tagset)
for tag in unique_tags:
    basic_model.add_states(states[tag])

# add the starting edges
for tag, tag_count in tag_starts.items():
    basic_model.add_transition(basic_model.start, states[tag], tag_count/len(data.training_set.X))

# add the ending edges
for tag, tag_count in tag_ends.items():
    basic_model.add_transition(states[tag], basic_model.end, tag_count/len(data.training_set.X))
    
# add the transitions
for bi_tag, tag_count in tag_bigrams.items():
    tag0 = bi_tag[0]
    tag1 = bi_tag[1]
    prob = tag_count/tag_unigrams[tag0]
    basic_model.add_transition(states[tag0], states[tag1], prob)

# finalize the model
basic_model.bake()

