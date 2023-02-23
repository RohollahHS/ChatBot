import torch
import torch.nn as nn
import random
import os

from config import args, DEVICE

from dataset import (
    SOS_token,
    batch2TrainData,
    voc,
    pairs,
    pairs_valid,
    save_dir,
)
from model import (
    embedding,
    encoder,
    decoder,
    encoder_optimizer,
    decoder_optimizer,
    encoder_scheduler,
    decoder_scheduler,
)
from utils import SaveBestModel, save_last_model


def maskNLLLoss(inp, target, mask):
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(DEVICE)
    return loss, nTotal.item()


def train(
    input_variable,
    lengths,
    target_variable,
    mask,
    max_target_len,
    encoder,
    decoder,
    embedding,
    encoder_optimizer,
    decoder_optimizer,
    batch_size,
    clip,
    max_length=args["max_length"],
):
    encoder.train()
    decoder.train()

    # Zero gradients
    # Ensure dropout layers are in train mode

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # Set DEVICE options
    input_variable = input_variable.to(DEVICE)
    target_variable = target_variable.to(DEVICE)
    mask = mask.to(DEVICE)
    # Lengths for rnn packing should always be on the cpu
    lengths = lengths.to("cpu")

    # Initialize variables
    loss = 0
    print_losses = []
    n_totals = 0

    # Forward pass through encoder
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

    # Create initial decoder input (start with SOS tokens for each sentence)
    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
    decoder_input = decoder_input.to(DEVICE)

    # Set initial decoder hidden state to the encoder's final hidden state
    if args["rnn_type"] == "LSTM":
        ht = encoder_hidden[0][: decoder.n_layers]
        ct = encoder_hidden[1][: decoder.n_layers]
        decoder_hidden = (ht, ct)
    elif args["rnn_type"] == "GRU":
        decoder_hidden = encoder_hidden[: decoder.n_layers]

    # Determine if we are using teacher forcing this iteration
    use_teacher_forcing = True if random.random() < args["teacher_forcing_ratio"] else False

    # Forward batch of sequences through decoder one time step at a time
    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # Teacher forcing: next input is current target
            decoder_input = target_variable[t].view(1, -1)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
    else:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # No teacher forcing: next input is decoder's own current output
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(DEVICE)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal

    # Perform backpropatation
    loss.backward()

    # Clip gradients: gradients are modified in place
    _ = nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # Adjust model weights
    encoder_optimizer.step()
    decoder_optimizer.step()

    encoder_scheduler.step()
    decoder_scheduler.step()

    return sum(print_losses) / n_totals


def trainIters(
    directory,
    voc,
    pairs,
    encoder,
    decoder,
    encoder_optimizer,
    decoder_optimizer,
    embedding,
    encoder_n_layers,
    decoder_n_layers,
    save_dir,
    n_iteration,
    batch_size,
    print_every,
    save_every,
    clip,
    corpus_name,
    loadfilename,
):

    # Load batches for each iteration
    training_batches = [
        batch2TrainData(voc, [random.choice(pairs) for _ in range(batch_size)])
        for _ in range(n_iteration)
    ]

    # Initializations
    start_iteration = 1
    print_loss = 0
    if args["train_mode"] == "continue":
        start_iteration = checkpoint["iteration"] + 1

    if not os.path.exists(directory):
        os.makedirs(directory)

    # Training loop
    for iteration in range(start_iteration, n_iteration + 1):
        training_batch = training_batches[iteration - 1]
        # Extract fields from batch
        input_variable, lengths, target_variable, mask, max_target_len = training_batch

        # Run a training iteration with batch
        loss = train(
            input_variable,
            lengths,
            target_variable,
            mask,
            max_target_len,
            encoder,
            decoder,
            embedding,
            encoder_optimizer,
            decoder_optimizer,
            batch_size,
            clip,
        )
        print_loss += loss

        with torch.no_grad():
            loss_valid = validIters()

        train_loss_txt = open(f"{args['out_dir']}/train_loss.txt", "a")
        valid_loss_txt = open(f"{args['out_dir']}/valid_loss.txt", "a")
        train_loss_txt.write(f"{iteration}: {loss:.4f} | ")
        valid_loss_txt.write(f"{iteration}: {loss_valid:.4f} | ")
        train_loss_txt.close()
        valid_loss_txt.close()

        train_loss_per_iteration.append(loss)
        valid_loss_per_iteration.append(loss_valid)

        # Print progress
        if (iteration >= args["start_save"]) and (loss_valid < save_best_model.best_valid):
            print_loss_avg = print_loss / print_every
            print(
                "| Iteration: {:4}/{} | Train loss: {:.4f} | Valid loss: {:.4f} | Save best model ... |".format(
                    iteration, n_iteration, print_loss_avg, loss_valid
                )
            )
            print(
                "+" + 22 * "-" + "+" + 20 * "-" + "+" + 20 * "-" + "+" + 21 * "-" + "+"
            )
            print_loss = 0
            save_best_model(
                loss_valid,
                iteration,
                encoder,
                decoder,
                encoder_optimizer,
                decoder_optimizer,
                voc,
                embedding,
                train_loss_per_iteration,
                valid_loss_per_iteration,
                encoder_scheduler,
                decoder_scheduler,
                directory,
            )
            save_last_model(
                iteration,
                encoder,
                decoder,
                encoder_optimizer,
                decoder_optimizer,
                voc,
                embedding,
                train_loss_per_iteration,
                valid_loss_per_iteration,
                encoder_scheduler,
                decoder_scheduler,
                directory,
            )
        else:
            print_loss_avg = print_loss / print_every
            print(
                "| Iteration: {:4}/{} | Train loss: {:.4f} | Valid loss: {:.4f} |".format(
                    iteration, n_iteration, print_loss_avg, loss_valid
                )
            )
            print("+" + 22 * "-" + "+" + 20 * "-" + "+" + 20 * "-" + "+")
            print_loss = 0

            save_last_model(
                iteration,
                encoder,
                decoder,
                encoder_optimizer,
                decoder_optimizer,
                voc,
                embedding,
                train_loss_per_iteration,
                valid_loss_per_iteration,
                encoder_scheduler,
                decoder_scheduler,
                directory,
            )


def validIters():
    batch_size_valid = 1
    encoder.eval()
    decoder.eval()
    print_loss = 0
    for i in range(len(pairs_valid)):
        valid_sample = [batch2TrainData(voc, pairs_valid[i : i + 1])]
        input_variable, lengths, target_variable, mask, max_target_len = valid_sample[0]
        print_loss = 0
        loss = valid(
            input_variable,
            lengths,
            target_variable,
            mask,
            max_target_len,
            encoder,
            decoder,
            batch_size_valid,
        )

        print_loss += loss

    return print_loss


def valid(
    input_variable,
    lengths,
    target_variable,
    mask,
    max_target_len,
    encoder,
    decoder,
    batch_size,
    max_length=args["max_length"],
):
    # Set DEVICE options
    input_variable = input_variable.to(DEVICE)
    target_variable = target_variable.to(DEVICE)
    mask = mask.to(DEVICE)
    # Lengths for rnn packing should always be on the cpu
    lengths = lengths.to("cpu")

    # Initialize variables
    loss = 0
    print_losses = []
    n_totals = 0

    # Forward pass through encoder
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

    # Create initial decoder input (start with SOS tokens for each sentence)
    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
    decoder_input = decoder_input.to(DEVICE)

    # Set initial decoder hidden state to the encoder's final hidden state
    if args["rnn_type"] == "LSTM":
        ht = encoder_hidden[0][: decoder.n_layers]
        ct = encoder_hidden[1][: decoder.n_layers]
        decoder_hidden = (ht, ct)
    elif args["rnn_type"] == "GRU":
        decoder_hidden = encoder_hidden[: decoder.n_layers]

    for t in range(max_target_len):
        decoder_output, decoder_hidden = decoder(
            decoder_input, decoder_hidden, encoder_outputs
        )
        # No teacher forcing: next input is decoder's own current output
        _, topi = decoder_output.topk(1)
        decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
        decoder_input = decoder_input.to(DEVICE)
        # Calculate and accumulate loss
        mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
        loss += mask_loss
        print_losses.append(mask_loss.item() * nTotal)
        n_totals += nTotal

    return sum(print_losses) / n_totals


if args["train_mode"] == "continue":
    last_model_dir = os.path.join(args["out_directories"], args["last_model_dir"])
    checkpoint = torch.load(
        os.path.join(last_model_dir, "last_model_checkpoint.tar"), map_location=DEVICE
    )
    encoder_sd = checkpoint["en"]
    decoder_sd = checkpoint["de"]
    encoder_optimizer_sd = checkpoint["en_opt"]
    decoder_optimizer_sd = checkpoint["de_opt"]
    embedding_sd = checkpoint["embedding"]
    encoder_scheduler_sd = checkpoint["encoder_scheduler"]
    decoder_scheduler_sd = checkpoint["decoder_scheduler"]

    save_best_model = SaveBestModel(
        best_valid=checkpoint["valid_loss_per_iteration"][-1]
    )
    train_loss_per_iteration = checkpoint["train_loss_per_iteration"]
    valid_loss_per_iteration = checkpoint["valid_loss_per_iteration"]

    voc.__dict__ = checkpoint["voc_dict"]
    embedding.load_state_dict(embedding_sd)

    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)

    encoder_optimizer.load_state_dict(encoder_optimizer_sd)
    decoder_optimizer.load_state_dict(decoder_optimizer_sd)

    encoder_scheduler.load_state_dict(encoder_scheduler_sd)
    decoder_scheduler.load_state_dict(decoder_scheduler_sd)

    print("\nContinuing Training!\n")

else:
    save_best_model = SaveBestModel()
    train_loss_per_iteration = []
    valid_loss_per_iteration = []

    train_loss_txt = open(f"{args['out_dir']}/train_loss.txt", "a")
    valid_loss_txt = open(f"{args['out_dir']}/valid_loss.txt", "a")

    train_loss_txt.write(f"\n{args['save_dir_name']}\n")
    valid_loss_txt.write(f"\n{args['save_dir_name']}\n")

    train_loss_txt.close()
    valid_loss_txt.close()
    print("Starting Training!\n")


# Run training iterations
trainIters(
    args["out_dir"],
    voc,
    pairs,
    encoder,
    decoder,
    encoder_optimizer,
    decoder_optimizer,
    embedding,
    args["encoder_n_layers"],
    args["decoder_n_layers"],
    save_dir,
    args["n_iteration"],
    args["batch_size"],
    args["print_every"],
    args["save_every"],
    args["clip"],
    args["corpus_name"],
    args["load_file_name"],
)
