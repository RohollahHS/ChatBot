import torch
import csv
import random
import re
import os
import unicodedata
import codecs
from io import open
import itertools
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import (
    MAX_LENGTH,
    MIN_COUNT,
    CORPUS_NAME,
    FILE_NAME,
    FILE_NAME_VALID,
    ALL_SETS,
    ALL_LABELS,
    ALL_LABELS_1
)


def printLines(file, n=10):
    with open(file, "rb") as datafile:
        lines = datafile.readlines()
    for line in lines[:n]:
        print(line)


# Splits each line of the file to create lines and conversations
def loadLinesAndConversations(fileName):
    questions = {}

    # just for train this block exucuted if the condition meet
    if (FILE_NAME in fileName) and ALL_SETS: 
        dirs = fileName.split("\\")
        df1 = pd.read_csv(f"{dirs[0]}/{dirs[1]}/WikiQA-train.tsv", delimiter="\t")
        df2 = pd.read_csv(f"{dirs[0]}/{dirs[1]}/WikiQA-dev.tsv", delimiter="\t")
        df3 = pd.read_csv(f"{dirs[0]}/{dirs[1]}/WikiQA-test.tsv", delimiter="\t")
        df = pd.concat([df1, df2, df3])

    else:
        df = pd.read_csv(fileName, delimiter="\t")

    # all questions and answers, wether its labels is 1 or 0
    if (ALL_LABELS==True) and (ALL_LABELS_1==False):
        i = 0
        for d in range(df.shape[0]):
            row = df.iloc[d]
            qa = {}
            qa["q"] = row["Question"]
            qa["a"] = row["Sentence"]
            questions[i] = qa
            i += 1
        
        return questions
    
    # all questions & answers with label 1, even questions that have multiple label 1'
    elif (ALL_LABELS_1==True) and (ALL_LABELS==False):
        i = 0
        for d in range(df.shape[0]):
            row = df.iloc[d]
            if row["Label"] == 1:
                qa = {}
                qa["q"] = row["Question"]
                qa["a"] = row["Sentence"]
                questions[i] = qa
                i += 1
        
        return questions

    # all questions with label 1, and for each question just one answer exists
    elif (ALL_LABELS_1==False) and (ALL_LABELS==False):
        for d in range(df.shape[0]):
            row = df.iloc[d]
            Id = row["QuestionID"]
            if (row["Label"] == 1) and (Id not in questions.keys()):
                qa = {}
                qa["q"] = row["Question"]
                qa["a"] = row["Sentence"]
                questions[Id] = qa
        
        return questions
    


# Extracts pairs of sentences from conversations
def extractSentencePairs(question_answer):
    qa_pairs = []
    for qa in question_answer.values():
        # Iterate over all the lines of the conversation
        inputLine = qa["q"]
        targetLine = qa["a"]
        # Filter wrong samples (if one of the lists is empty)
        if inputLine and targetLine:
            qa_pairs.append([inputLine, targetLine])
    return qa_pairs


# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token


class Voc:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3  # Count SOS, EOS, PAD

    def addSentence(self, sentence):
        for word in sentence.split(" "):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    # Remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        # counts = list(self.word2count.values())
        # plt.hist(counts, range=[1, 20], bins=20)

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print(
            "keep_words {} / {} = {:.4f}".format(
                len(keep_words),
                len(self.word2index),
                len(keep_words) / len(self.word2index),
            )
        )

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3  # Count default tokens

        for word in keep_words:
            self.addWord(word)


# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    )


# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s


# Read query/response pairs and return a voc object
def readVocs(datafile, corpus_name=CORPUS_NAME):
    # print("Reading lines...")
    # Read the file and split into lines
    lines = open(datafile, encoding="utf-8").read().strip().split("\n")
    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split("\t")] for l in lines]
    voc = Voc(corpus_name)
    return voc, pairs


# Returns True iff both sentences in a pair 'p' are under the MAX_LENGTH threshold
def filterPair(p):
    # Input sequences need to preserve the last word for EOS token
    return len(p[0].split(" ")) < MAX_LENGTH and len(p[1].split(" ")) < MAX_LENGTH


# Filter pairs using filterPair condition
def filterPairs(pairs):
    # ql = []
    # al = []
    # for p in pairs:
    #     ql.append(len(p[0].split()))
    #     al.append(len(p[1].split()))
    # fig, axs = plt.subplots(2, 1, sharey=True, tight_layout=True)
    # axs[0].hist(ql, bins=10)
    # axs[1].hist(al, bins=20)
    # plt.show()

    # We can set the number of bins with the *bins* keyword argument.
    return [pair for pair in pairs if filterPair(pair)]


# Using the functions defined above, return a populated voc object and pairs list
def loadPrepareData(corpus, corpus_name, datafile, save_dir):
    # print("Start preparing training data ...")
    voc, pairs = readVocs(datafile, corpus_name)
    print("Read {!s} sentence pairs for train".format(len(pairs)))
    pairs = filterPairs(pairs)
    print(
        "Trimmed to {!s} sentence pairs for train with MAX_LENGTH {}".format(
            len(pairs), MAX_LENGTH
        )
    )
    # print("Counting words...")
    for pair in pairs:
        voc.addSentence(pair[0])
        voc.addSentence(pair[1])
    print("Counted words:", voc.num_words)
    return voc, pairs


# Using the functions defined above, return a populated voc object and pairs list
def loadPrepareDataValid(corpus, corpus_name, datafile, save_dir):
    # print("Start preparing training data ...")
    _, pairs = readVocs(datafile, corpus_name)
    print("Read {!s} sentence pairs for validation".format(len(pairs)))
    pairs = filterPairs(pairs)
    print(
        "Trimmed to {!s} sentence pairs for validation with MAX_LENGTH {}".format(
            len(pairs), MAX_LENGTH
        )
    )
    return pairs


def trimRareWords(voc, pairs, MIN_COUNT):
    # Trim words used under the MIN_COUNT from the voc
    voc.trim(MIN_COUNT)
    # Filter out pairs with trimmed words
    keep_pairs = []
    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True
        # Check input sentence
        for word in input_sentence.split(" "):
            if word not in voc.word2index:
                keep_input = False
                break
        # Check output sentence
        for word in output_sentence.split(" "):
            if word not in voc.word2index:
                keep_output = False
                break

        # Only keep pairs that do not contain trimmed word(s) in their input or output sentence
        if keep_input and keep_output:
            keep_pairs.append(pair)

    print(
        "Trimmed from {} pairs to {}, {:.4f} of total for train with MIN_COUNT {}".format(
            len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs), MIN_COUNT
        )
    )
    return keep_pairs


def trimRareWordsValid(voc, pairs, MIN_COUNT):
    # Filter out pairs with trimmed words
    keep_pairs = []
    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True
        # Check input sentence
        for word in input_sentence.split(" "):
            if word not in voc.word2index:
                keep_input = False
                break
        # Check output sentence
        for word in output_sentence.split(" "):
            if word not in voc.word2index:
                keep_output = False
                break

        # Only keep pairs that do not contain trimmed word(s) in their input or output sentence
        if keep_input and keep_output:
            keep_pairs.append(pair)

    print(
        "Trimmed from {} pairs to {}, {:.4f} of total for validation with MIN_COUNT {}".format(
            len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs), MIN_COUNT
        )
    )
    return keep_pairs


def indexesFromSentence(voc, sentence):
    return [voc.word2index[word] for word in sentence.split(" ")] + [EOS_token]


def zeroPadding(l, fillvalue=PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))


def binaryMatrix(l, value=PAD_token):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m


# Returns padded input sequence tensor and lengths
def inputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths


# Returns padded target sequence tensor, padding mask, and max target length
def outputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = torch.BoolTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len


# Returns all items for a given batch of pairs
def batch2TrainData(voc, pair_batch):
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = inputVar(input_batch, voc)
    output, mask, max_target_len = outputVar(output_batch, voc)
    return inp, lengths, output, mask, max_target_len


# corpus_name = "movie-corpus"
corpus = os.path.join("data", CORPUS_NAME)

# printLines(os.path.join(corpus, FILE_NAME))
# printLines(os.path.join(corpus, FILE_NAME_VALID))

# Define path to new file
datafile = os.path.join(corpus, "formatted_movie_lines.txt")
datafile_valid = os.path.join(corpus, "formatted_movie_lines_valid.txt")

delimiter = "\t"
# Unescape the delimiter
delimiter = str(codecs.decode(delimiter, "unicode_escape"))

# Initialize lines dict and conversations dict
lines = {}
conversations = {}
# Load lines and conversations
# print("\nProcessing corpus into lines and conversations...")
questions = loadLinesAndConversations(os.path.join(corpus, FILE_NAME))
questions_valid = loadLinesAndConversations(os.path.join(corpus, FILE_NAME_VALID))

# Write new csv file
# print("\nWriting newly formatted file...")

with open(datafile, "w", encoding="utf-8") as outputfile:
    writer = csv.writer(outputfile, delimiter=delimiter, lineterminator="\n")
    for pair in extractSentencePairs(questions):
        writer.writerow(pair)

with open(datafile_valid, "w", encoding="utf-8") as outputfile:
    writer = csv.writer(outputfile, delimiter=delimiter, lineterminator="\n")
    for pair in extractSentencePairs(questions_valid):
        writer.writerow(pair)


# Print a sample of lines
# print("\nSample lines from file:")
# printLines(datafile)
# printLines(datafile_valid)


# Load/Assemble voc and pairs
save_dir = os.path.join("data", "save")
voc, pairs = loadPrepareData(corpus, CORPUS_NAME, datafile, save_dir)
pairs_valid = loadPrepareDataValid(corpus, CORPUS_NAME, datafile_valid, save_dir)
# Print some pairs to validate
# print("\npairs:")
# for pair in pairs[:10]:
#     print(pair)


# Trim voc and pairs
pairs = trimRareWords(voc, pairs, MIN_COUNT)
pairs_valid = trimRareWordsValid(voc, pairs_valid, MIN_COUNT)

print("\nNumber of train samples: ", len(pairs))
print("Number of valid samples: ", len(pairs_valid))


# # Example for validation
# small_batch_size = 5
# batches = batch2TrainData(voc, [random.choice(pairs) for _ in range(small_batch_size)])
# input_variable, lengths, target_variable, mask, max_target_len = batches

# print("input_variable:", input_variable)
# print("lengths:", lengths)
# print("target_variable:", target_variable)
# print("mask:", mask)
# print("max_target_len:", max_target_len)
