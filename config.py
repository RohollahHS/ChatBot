import torch
import argparse
import os
import datetime

parser = argparse.ArgumentParser()

# Model options
parser.add_argument("--rnn_type", default="GRU", type=str, choices=["GRU", "LSTM"])
parser.add_argument("--attn_model", default="dot", type=str, choices=["dot", "general", "concat"])
parser.add_argument("--hidden_size", default=512, type=int)
parser.add_argument("--encoder_n_layers", default=4, type=int)
parser.add_argument("--decoder_n_layers", default=4, type=int)
parser.add_argument("--dropout", default=0.1, type=float)

# Training options
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--lr", default=0.0001, type=float)
parser.add_argument("--schedule", nargs="*", default=[2500], type=int)
parser.add_argument("--lr_decay_ratio", default=0.1, type=float)
parser.add_argument("--n_iteration", default=3000, type=int)
parser.add_argument("--weight_decay", default=5e-5, type=float)
parser.add_argument("--clip", default=50.0, type=float)
parser.add_argument("--teacher_forcing_ratio", default=0.8, type=float)
parser.add_argument("--decoder_learning_ratio", default=5.0, type=float)
parser.add_argument("--train_mode", default="start", choices=["start", "continue"], type=str)
parser.add_argument("--last_model_dir", default="2023-02-23_09-40-20", type=str)

parser.add_argument("--print_every", default=1, type=int)
parser.add_argument("--save_every", default=1, type=int)
parser.add_argument("--load_file_name", default=None)
parser.add_argument("--valid_every", default=1, type=int)
parser.add_argument("--start_save", default=500, type=int)

# Dataset options
parser.add_argument(
    "--dataset_type",
    default="all_labels_distinct",
    type=str,
    choices=[
        "all_labels_distinct",
        "all_labels",
        "all_labels_1",
        "all_labels_1_distinct",
    ],
    help="if all_labels_distinct: all distinct questions will be seleted, each question just has one answer"
    + "if all_labels: all questions will be seleted, each question has multiple answer"
    + "if all_labels_1: all questions with labels 1 will be selected, each question may has multiple answer"
    + "if all_labels_1_distinct: all distinct questions with labels 1 will be selected, each question has one answer",
)
parser.add_argument("--all_sets", default=True, type=bool)
parser.add_argument("--max_length", default=40, type=int)
parser.add_argument("--min_count", default=1, type=int)
parser.add_argument("--corpus_name", default="WikiQA", type=str)
parser.add_argument("--file_name", default="WikiQA-train.tsv", type=str)
parser.add_argument("--file_name_valid", default="WikiQA-dev.tsv", type=str)
parser.add_argument("--file_name_test", default="WikiQA-test.tsv", type=str)
parser.add_argument("--out_directories", default="drive/MyDrive/University/Big_Data/HW3", type=str)
parser.add_argument("--note", default="", type=str)

args = vars(parser.parse_args())

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

date_and_time = datetime.datetime.now()
date_and_time = str(date_and_time).split()
date = date_and_time[0]
time = date_and_time[1].split(".")[0].replace(":", "-")
date_and_time = date + "_" + time

args["out_dir"] = os.path.join(args["out_directories"], date_and_time)
args["save_dir_name"] = date_and_time

if not os.path.exists(os.path.join(args["out_dir"])):
    os.makedirs(os.path.join(args["out_dir"]), exist_ok=True)

model_details = open(os.path.join(args["out_dir"], "model_details.txt"), "w")
print("Model details:")
for k, v in args.items():
    model_details.write(f"{k:22s}: {v}\n")
    model_details.write(81*"-")
    model_details.write("\n")
    print(f"{k:22s}: {v}")
    print(81*"-")
model_details.close()
