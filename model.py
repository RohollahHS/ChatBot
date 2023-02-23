import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from config import args, DEVICE

from dataset import (
    voc,
)


class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0, rnn_type=args["rnn_type"]):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding

        # Initialize RNN; the input_size and hidden_size params are both set to 'hidden_size'
        #   because our input size is a word embedding with number of features == hidden_size
        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(
                hidden_size,
                hidden_size,
                n_layers,
                dropout=(0 if n_layers == 1 else dropout),
                bidirectional=True,
            )
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(
                hidden_size,
                hidden_size,
                n_layers,
                dropout=(0 if n_layers == 1 else dropout),
                bidirectional=True,
            )

    def forward(self, input_seq, input_lengths, hidden=None):
        # Convert word indexes to embeddings
        embedded = self.embedding(input_seq)
        # Pack padded batch of sequences for RNN module
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        
        # Forward pass through RNN
        outputs, hidden = self.rnn(packed, hidden)

        # Unpack padding
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        # Sum bidirectional RNN outputs
        outputs = (outputs[:, :, : self.hidden_size] + 
                           outputs[:, :, self.hidden_size :])
        # Return output and final hidden state
        return outputs, hidden


# Luong attention layer
class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ["dot", "general", "concat"]:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == "general":
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == "concat":
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(
            torch.cat(
                (hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2
            )
        ).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        # Calculate the attention weights (energies) based on the given method
        if self.method == "general":
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == "concat":
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == "dot":
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # Transpose max_length and batch_size dimensions
        attn_energies = attn_energies.t()

        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)


class LuongAttnDecoderRNN(nn.Module):
    def __init__(
        self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1, rnn_type=args["rnn_type"]
    ):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(
                hidden_size,
                hidden_size,
                n_layers,
                dropout=(0 if n_layers == 1 else dropout),
            )
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(
                hidden_size,
                hidden_size,
                n_layers,
                dropout=(0 if n_layers == 1 else dropout),
            )

        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        # Note: we run this one step (word) at a time
        # Get embedding of current input word
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)

        # Forward through unidirectional RNN
        rnn_output, hidden = self.rnn(embedded, last_hidden)

        # Calculate attention weights from the current RNN output
        attn_weights = self.attn(rnn_output, encoder_outputs)
        # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        # Concatenate weighted context vector and RNN output using Luong eq. 5
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        # Predict next word using Luong eq. 6
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)
        # Return output and final hidden state
        return output, hidden


# Set checkpoint to load from; set to None if starting from scratch


# print("Building encoder and decoder ...")
# Initialize word embeddings
embedding = nn.Embedding(voc.num_words, args["hidden_size"])

# Initialize encoder & decoder models
encoder = EncoderRNN(args["hidden_size"], embedding, args["encoder_n_layers"], args["dropout"])
decoder = LuongAttnDecoderRNN(
    args["attn_model"], embedding, args["hidden_size"], voc.num_words, args["decoder_n_layers"], args["dropout"]
)

# Use appropriate DEVICE
encoder = encoder.to(DEVICE)
decoder = decoder.to(DEVICE)
# print("Models built and ready to go!")


# Initialize optimizers
# print("Building optimizers ...")
encoder_optimizer = optim.Adam(
    encoder.parameters(), lr=args["lr"], weight_decay=args["weight_decay"]
)
decoder_optimizer = optim.Adam(
    decoder.parameters(),
    lr=args["lr"] * args["decoder_learning_ratio"],
    weight_decay=args["weight_decay"],
)

encoder_scheduler = optim.lr_scheduler.MultiStepLR(
    encoder_optimizer, milestones=args["schedule"], gamma=args["lr_decay_ratio"]
)
decoder_scheduler = optim.lr_scheduler.MultiStepLR(
    decoder_optimizer, milestones=args["schedule"], gamma=args["lr_decay_ratio"]
)

print()
print("Nubmer of parameters for Encoder:")
n = 0
for p in encoder.parameters():
    n += p.numel()
print(n)

print("Nubmer of parameters for Decoder:")
n = 0
for p in decoder.parameters():
    n += p.numel()
print(n)
print()

# If you have cuda, configure cuda to call
for state in encoder_optimizer.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.cuda()

for state in decoder_optimizer.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.cuda()
