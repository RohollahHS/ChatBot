import torch
import torch.nn as nn
import random
import os

from config import (
    DEVICE,
    MAX_LENGTH,
    CLIP,
    TEACHER_FORCING_RATIO,
    N_EPOCH,
    MODEL_NAME,
    HIDDEN_SIZE,
    ENCODER_N_LAYERS,
    DECODER_N_LAYERS,
    BATCH_SIZE,
    CORPUS_NAME,
    RNN_TYPE,
    OUT_DIR,
    TRAIN_MODE
)
from dataset import (
    SOS_token,
    batch2TrainData,
    voc,
    # train_loader,
    # valid_loader,
    pairs,
    pairs_valid
)
from model import (
    embedding,
    encoder,
    decoder,
    encoder_optimizer,
    decoder_optimizer,
    encoder_scheduler,
    decoder_scheduler
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
    max_length=MAX_LENGTH,
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
    batch_size = len(lengths)
    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
    decoder_input = decoder_input.to(DEVICE)

    # Set initial decoder hidden state to the encoder's final hidden state
    if RNN_TYPE == "LSTM":
        hn = encoder_hidden[0][: decoder.n_layers]
        cn = encoder_hidden[1][: decoder.n_layers]
        decoder_hidden = (hn, cn)
    elif RNN_TYPE == "GRU":
        decoder_hidden = encoder_hidden[: decoder.n_layers]
    

    # Determine if we are using teacher forcing this epoch
    use_teacher_forcing = True if random.random() < TEACHER_FORCING_RATIO else False

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

def validIters():
    encoder.eval()
    decoder.eval()

    print_loss = 0
    batches = [pairs_valid[i:i+BATCH_SIZE] for i in range(0, len(pairs_valid), BATCH_SIZE)]
    for batch in batches:
        valid_batch = batch2TrainData(voc, batch)
        input_variable, lengths, target_variable, mask, max_target_len = valid_batch
        
        loss = valid(
            input_variable,
            lengths,
            target_variable,
            mask,
            max_target_len,
            encoder,
            decoder,
        )

        print_loss += loss
    
    if len(batches) == 0:
        return 0
    
    return print_loss / len(batches)


def valid(
    input_variable,
    lengths,
    target_variable,
    mask,
    max_target_len,
    encoder,
    decoder,
    max_length=MAX_LENGTH,
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
    batch_size = len(lengths)
    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
    decoder_input = decoder_input.to(DEVICE)

    # Set initial decoder hidden state to the encoder's final hidden state
    if RNN_TYPE == "LSTM":
        hn = encoder_hidden[0][: decoder.n_layers]
        cn = encoder_hidden[1][: decoder.n_layers]
        decoder_hidden = (hn, cn)
    elif RNN_TYPE == "GRU":
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


def trainIters(
    model_name,
    voc,
    # train_loader,
    encoder,
    decoder,
    encoder_optimizer,
    decoder_optimizer,
    embedding,
    encoder_n_layers,
    decoder_n_layers,
    n_epoch,
    batch_size,
    clip,
    corpus_name,
):

    # Initializations
    start_epoch = 1
    if TRAIN_MODE == "continue":
        start_epoch = checkpoint["epoch"] + 1

    directory = os.path.join(
        OUT_DIR,
        model_name.replace(', ', '-').replace(': ', '_').replace(',', '').strip(),
    )
    if not os.path.exists(directory):
        os.makedirs(directory)

    print_loss = 0
    for epoch in range(start_epoch, n_epoch + 1):
        random.shuffle(pairs)
        batches = [pairs[i:i+BATCH_SIZE] for i in range(0, len(pairs), BATCH_SIZE)]
        for batch in batches:
            training_batch = batch2TrainData(voc, batch)
            input_variable, lengths, target_variable, mask, max_target_len = training_batch

            # Run a training epoch with batch
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

        print_loss_avg = print_loss / len(batches)

        with torch.no_grad():
            loss_valid = validIters()

        train_loss_txt = open(f'{OUT_DIR}/train_loss.txt', 'a')
        valid_loss_txt = open(f'{OUT_DIR}/valid_loss.txt', 'a')
        train_loss_txt.write(f'{epoch}: {loss:.4f} | ')
        valid_loss_txt.write(f'{epoch}: {loss_valid:.4f} | ')
        train_loss_txt.close()
        valid_loss_txt.close()
        
        train_loss_per_epoch.append(loss)
        valid_loss_per_epoch.append(loss_valid)

        # Print progress
        if loss_valid < save_best_model.best_valid:
            print(
                "| Epoch: {:3}/{:<3} | Train loss: {:.4f} | Valid loss: {:.4f} | Save best model ... |".format(
                    epoch, n_epoch, print_loss_avg, loss_valid
                )
            )
            print(83*"-")
            print_loss = 0

        else:
            print(
                "| Epoch: {:3}/{:<3} | Train loss: {:.4f} | Valid loss: {:.4f} |".format(
                    epoch, n_epoch, print_loss_avg, loss_valid
                )
            )
            print(61*"-")
            print_loss = 0
        

        # Save checkpoint
        save_best_model(
                loss_valid,
                epoch,
                encoder,
                decoder,
                encoder_optimizer,
                decoder_optimizer,
                voc,
                embedding,
                train_loss_per_epoch,
                valid_loss_per_epoch,
                encoder_scheduler,
                decoder_scheduler,
                directory
            )

        # saving last model
        save_last_model(
            epoch,
            encoder,
            decoder,
            encoder_optimizer,
            decoder_optimizer,
            voc,
            embedding,
            train_loss_per_epoch,
            valid_loss_per_epoch,
            encoder_scheduler,
            decoder_scheduler,
            directory
        )


# Load model if a LOADFILENAME is provided
if TRAIN_MODE == "continue":
    save_dir = os.path.join(
        OUT_DIR,
        MODEL_NAME.replace(', ', '-').replace(': ', '_').replace(',', '').strip(),
    )
    # If loading on same machine the model was trained on
    checkpoint = torch.load(os.path.join(save_dir, "last_model_checkpoint.tar"), map_location=DEVICE)
    # If loading a model trained on GPU to CPU
    # checkpoint = torch.load(LOADFILENAME, map_location=torch.DEVICE('cpu'))
    encoder_sd = checkpoint["en"]
    decoder_sd = checkpoint["de"]
    encoder_optimizer_sd = checkpoint["en_opt"]
    decoder_optimizer_sd = checkpoint["de_opt"]
    embedding_sd = checkpoint["embedding"]
    encoder_scheduler_sd = checkpoint["encoder_scheduler"]
    decoder_scheduler_sd = checkpoint["decoder_scheduler"]

    save_best_model = SaveBestModel(best_valid=checkpoint["valid_loss_per_epoch"][-1])
    train_loss_per_epoch = checkpoint["train_loss_per_epoch"]
    valid_loss_per_epoch = checkpoint["valid_loss_per_epoch"]

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
    train_loss_per_epoch = []
    valid_loss_per_epoch = []

    train_loss_txt = open(f'{OUT_DIR}/train_loss.txt', 'a')
    valid_loss_txt = open(f'{OUT_DIR}/valid_loss.txt', 'a')

    train_loss_txt.write(f'\n{MODEL_NAME}\n')
    valid_loss_txt.write(f'\n{MODEL_NAME}\n')

    train_loss_txt.close()
    valid_loss_txt.close()
    print("Starting Training!\n")


# Run training epochs
trainIters(
    MODEL_NAME,
    voc,
    # train_loader,
    encoder,
    decoder,
    encoder_optimizer,
    decoder_optimizer,
    embedding,
    ENCODER_N_LAYERS,
    DECODER_N_LAYERS,
    N_EPOCH,
    BATCH_SIZE,
    CLIP,
    CORPUS_NAME,
)
