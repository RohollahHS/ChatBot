import torch
import torch.nn as nn
import random
import os


from config import (
    DEVICE,
    MAX_LENGTH,
    CLIP,
    TEACHER_FORCING_RATIO,
    N_ITERATION,
    PRINT_EVERY,
    SAVE_EVERY,
    MODEL_NAME,
    HIDDEN_SIZE,
    ENCODER_N_LAYERS,
    DECODER_N_LAYERS,
    BATCH_SIZE,
    LOADFILENAME,
    CORPUS_NAME,
    VALID_EVERY,
    RNN_TYPE
)
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
)
from eval import searcher, evaluate


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
    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
    decoder_input = decoder_input.to(DEVICE)

    # Set initial decoder hidden state to the encoder's final hidden state
    if RNN_TYPE == "LSTM":
        hn = encoder_hidden[0][: decoder.n_layers]
        cn = encoder_hidden[1][: decoder.n_layers]
        decoder_hidden = (hn, cn)
    elif RNN_TYPE == "GRU":
        decoder_hidden = encoder_hidden[: decoder.n_layers]
    

    # Determine if we are using teacher forcing this iteration
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

    return sum(print_losses) / n_totals


def trainIters(
    model_name,
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
    
    print("Initializing ...")

    # Load batches for each iteration
    training_batches = [
        batch2TrainData(voc, [random.choice(pairs) for _ in range(batch_size)])
        for _ in range(n_iteration)
    ]

    # Initializations
    start_iteration = 1
    print_loss = 0
    if loadfilename:
        start_iteration = checkpoint["iteration"] + 1

    # Training loop
    print("Training...")
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

        if iteration % VALID_EVERY == 0:
            with torch.no_grad():
                loss_valid = validation()

        # Print progress
        if iteration % print_every == 0:
            print_loss_avg = print_loss / print_every
            print(
                "Iteration: {}; Percent complete: {:.1f}%; Train loss: {:.4f}; Valid loss: {:.4f}".format(
                    iteration, iteration / n_iteration * 100, print_loss_avg, loss_valid
                )
            )
            print_loss = 0
        

        # Save checkpoint
        if iteration % save_every == 0:
            directory = os.path.join(
                save_dir,
                model_name,
                corpus_name,
                "{}-{}_{}".format(encoder_n_layers, decoder_n_layers, HIDDEN_SIZE),
            )
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save(
                {
                    "iteration": iteration,
                    "en": encoder.state_dict(),
                    "de": decoder.state_dict(),
                    "en_opt": encoder_optimizer.state_dict(),
                    "de_opt": decoder_optimizer.state_dict(),
                    "loss": loss,
                    "voc_dict": voc.__dict__,
                    "embedding": embedding.state_dict(),
                },
                os.path.join(directory, "{}_{}.tar".format(iteration, "checkpoint")),
            )


def validation():
    batch_size_valid = 1
    encoder.eval()
    decoder.eval()
    for i in range(len(pairs_valid)):
        valid_sample = [batch2TrainData(voc, pairs_valid[i:i+1])]
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





checkpoint_iter = 4000
# LOADFILENAME = os.path.join(save_dir, MODEL_NAME, CORPUS_NAME,
#                            '{}-{}_{}'.format(ENCODER_N_LAYERS, DECODER_N_LAYERS, HIDDEN_SIZE),
#                            '{}_checkpoint.tar'.format(checkpoint_iter))

# Load model if a LOADFILENAME is provided
if LOADFILENAME:
    # If loading on same machine the model was trained on
    checkpoint = torch.load(LOADFILENAME)
    # If loading a model trained on GPU to CPU
    # checkpoint = torch.load(LOADFILENAME, map_location=torch.DEVICE('cpu'))
    encoder_sd = checkpoint["en"]
    decoder_sd = checkpoint["de"]
    encoder_optimizer_sd = checkpoint["en_opt"]
    decoder_optimizer_sd = checkpoint["de_opt"]
    embedding_sd = checkpoint["embedding"]
    voc.__dict__ = checkpoint["voc_dict"]

    embedding.load_state_dict(embedding_sd)
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)
    encoder_optimizer.load_state_dict(encoder_optimizer_sd)
    decoder_optimizer.load_state_dict(decoder_optimizer_sd)


# Run training iterations
print("Starting Training!")
trainIters(
    MODEL_NAME,
    voc,
    pairs,
    encoder,
    decoder,
    encoder_optimizer,
    decoder_optimizer,
    embedding,
    ENCODER_N_LAYERS,
    DECODER_N_LAYERS,
    save_dir,
    N_ITERATION,
    BATCH_SIZE,
    PRINT_EVERY,
    SAVE_EVERY,
    CLIP,
    CORPUS_NAME,
    LOADFILENAME,
)
