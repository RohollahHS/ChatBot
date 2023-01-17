import torch
import argparse

"""
!python train.py \
--rnn_type \
--attn_model \
--hidden_size \
--encoder_n_layers \
--decoder_n_layers \
--dropout \
--batch_size \
--lr \
--n_iteration \
--weight_decay \
--clip \
--teacher_forcing_ratio \
--decoder_learning_ratio \
--print_every \
--save_every \
--load_file_name \
--valid_every \
--max_length \
--min_count \
--corpus_name \
--file_name \
--file_name_valid \
--out_dir \
--note
"""

parser = argparse.ArgumentParser()

# Model options
parser.add_argument('--rnn_type', default="LSTM", type=str, choices=["GRU", "LSTM"])
parser.add_argument('--attn_model', default="dot", type=str, choices=["dot", "general", "concat"])
parser.add_argument('--hidden_size', default=512, type=int)
parser.add_argument('--encoder_n_layers', default=2, type=int)
parser.add_argument('--decoder_n_layers', default=2, type=int)
parser.add_argument('--dropout', default=0.1, type=float)

# Training options
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--lr', default=0.0001, type=float)
parser.add_argument('--schedule', nargs='*', default=[1000, 2000, 3000], type=int)
parser.add_argument('--lr_decay_ratio', default=0.1, type=float)
parser.add_argument('--n_iteration', default=1000, type=int)
parser.add_argument('--weight_decay', default=0.0005, type=float)
parser.add_argument('--clip', default=50.0, type=float)
parser.add_argument('--teacher_forcing_ratio', default=1.0, type=float)
parser.add_argument('--decoder_learning_ratio', default=5.0, type=float)

parser.add_argument('--print_every', default=1, type=int)
parser.add_argument('--save_every', default=1, type=int)
parser.add_argument('--load_file_name', default=None)
parser.add_argument('--valid_every', default=1, type=int)
parser.add_argument('--start_save', default=1, type=int)

# Dataset options
parser.add_argument('--all_labels', default=False, type=bool)
parser.add_argument('--all_labels_1', default=True, type=bool)
parser.add_argument('--max_length', default=80, type=int)
parser.add_argument('--min_count', default=1, type=int)
parser.add_argument('--corpus_name', default="WikiQA", type=str)
parser.add_argument('--all_sets', default=False, type=bool)
parser.add_argument('--file_name', default="WikiQA-train.tsv", type=str)
parser.add_argument('--file_name_valid', default="WikiQA-dev.tsv", type=str)
parser.add_argument('--out_dir', default="outputs", type=str)
parser.add_argument('--note', default="", type=str)

args = parser.parse_args()

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

# Configure models
RNN_TYPE = args.rnn_type
ATTN_MODEL = args.attn_model
HIDDEN_SIZE = args.hidden_size
ENCODER_N_LAYERS = args.encoder_n_layers
DECODER_N_LAYERS = args.decoder_n_layers
DROPOUT = args.dropout

# Configure training/optimization
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.lr
SCHEDULER_LIMESTONES = args.schedule
LR_DECAY_RATIO = args.lr_decay_ratio
WIEGHT_DECAY = args.weight_decay
N_ITERATION = args.n_iteration
CLIP = args.clip
TEACHER_FORCING_RATIO = args.teacher_forcing_ratio
DECODER_LEARNING_RATIO = args.decoder_learning_ratio
PRINT_EVERY = args.print_every
SAVE_EVERY = args.save_every
LOADFILENAME = args.load_file_name
VALID_EVERY = args.valid_every
START_SAVE = args.start_save

##### WikiQA Dataset
ALL_LABELS = args.all_labels
ALL_LABELS_1 = args.all_labels_1
MAX_LENGTH = args.max_length  # Maximum sentence length to consider
MIN_COUNT = args.min_count  # Minimum word count threshold for trimming
CORPUS_NAME = args.corpus_name
FILE_NAME = args.file_name
FILE_NAME_VALID = args.file_name_valid
OUT_DIR = args.out_dir
ALL_SETS = args.all_sets

##### Movie-Courpus Dataset
# CORPUS_NAME = "movie-corpus"
# # FILE_NAME = "utterances.jsonl"
# FILE_NAME = "utterances_1000_sample.jsonl"
# FILE_NAME_VALID = "utterances_valid.jsonl"

MODEL_NAME = f"ALL_LABELS: {ALL_LABELS}, ALL_LABELS_1: {ALL_LABELS_1}, MAX_LENGTH: {MAX_LENGTH}, MIN_COUNT: {MIN_COUNT}, ATTN: {ATTN_MODEL}, RNN: {RNN_TYPE}, HIDDEN: {HIDDEN_SIZE}, N_LAYERS: {ENCODER_N_LAYERS}, BATCH: {BATCH_SIZE}, TEACHER_RATIO: {TEACHER_FORCING_RATIO}, LR: {LEARNING_RATE}, DEC_LR_RATIO: {DECODER_LEARNING_RATIO}, N_ITERATION: {N_ITERATION}, {args.note}".strip()

details = MODEL_NAME.split(',')
print('\nModel details:')
for d in details:
    print(d.strip())
