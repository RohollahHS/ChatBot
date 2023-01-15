import torch

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")


# Configure models
MODEL_NAME = "cb_model"
ATTN_MODEL = "dot"
# ATTN_MODEL = 'general'
# ATTN_MODEL = 'concat'
HIDDEN_SIZE = 500
ENCODER_N_LAYERS = 2
DECODER_N_LAYERS = 2
DROPOUT = 0.1
BATCH_SIZE = 3

# Configure training/optimization
CLIP = 50.0
TEACHER_FORCING_RATIO = 1.0
LEARNING_RATE = 0.0001
DECODER_LEARNING_RATIO = 5.0
N_ITERATION = 500
PRINT_EVERY = 1
SAVE_EVERY = 500
LOADFILENAME = None
VALID_EVERY = 1


##### WikiQA Dataset
MAX_LENGTH = 80  # Maximum sentence length to consider
MIN_COUNT = 1  # Minimum word count threshold for trimming
CORPUS_NAME = "WikiQA"
FILE_NAME = "WikiQA-train.tsv"
FILE_NAME_VALID = "WikiQA-dev.tsv"


##### Movie-Courpus Dataset
# CORPUS_NAME = "movie-corpus"
# # FILE_NAME = "utterances.jsonl"
# FILE_NAME = "utterances_1000_sample.jsonl"
# FILE_NAME_VALID = "utterances_valid.jsonl"