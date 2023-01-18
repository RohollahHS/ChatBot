import numpy as np
import torch
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import os
import argparse
import sys

print(sys.argv[0])

if ("utils.py" in sys.argv[0]) or (len(sys.argv[0]) == 0):
    parser = argparse.ArgumentParser()

    parser.add_argument("--index", default=0, type=int)
    parser.add_argument("--out_dir", default="outputs", type=str)
    parser.add_argument("--best", default=False, type=bool)

    args = parser.parse_args()

    LOADFILENAME = args.out_dir
    best = args.best

    USE_CUDA = torch.cuda.is_available()
    DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

    files = os.listdir(LOADFILENAME)
    i = 0
    print("Saved models are:\n")
    for l in files:
        if "txt" not in l:
            print(f"{i}: {l}\n")
            i += 1
    i = args.index

    model_details = files[i].split(",")

    details = {}
    print("\nModel details:")
    for d in model_details:
        if len(d) != 0:
            k, v = d.split()
            print(f"{k}: {v}")
            details[k] = v

    all_models = os.listdir(os.path.join(LOADFILENAME, files[i]))
    if best:
        check = torch.load(
            os.path.join(LOADFILENAME, files[i], "best_model_checkpoint.tar"),
            map_location=DEVICE
        )
    else:
        check = torch.load(
            os.path.join(LOADFILENAME, files[i], "last_model_checkpoint.tar"),
            map_location=DEVICE
        )



def display_train_loss():
    plt.figure(figsize=(12, 7))
    plt.plot(check['train_loss_per_iteration'], linewidth=3)
    plt.ylabel('Loss')
    plt.xlabel('Iteration')
    plt.show()

def display_valid_loss():
    plt.figure(figsize=(12, 7))
    plt.plot(check['valid_loss_per_iteration'], linewidth=3)
    plt.ylabel('Loss')
    plt.xlabel('Iteration')
    plt.show()



