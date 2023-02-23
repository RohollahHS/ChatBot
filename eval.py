import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import re
import unicodedata
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--out_directories", default="drive/MyDrive/University/Big_Data/HW3", type=str
)
parser.add_argument("--out_dir", default="2023-02-23_11-01-49", type=str)

args = vars(parser.parse_args())

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

last_model_dir = os.path.join(args["out_directories"], args["out_dir"])
checkpoint = torch.load(
    os.path.join(last_model_dir, "last_model_checkpoint.tar"), map_location=DEVICE
)

with open(os.path.join(last_model_dir, "model_details.txt"), "r") as f:
    lines = f.readlines()

model_details = {}
print("Model details:")
for line in lines:
    if line[0] != "-":
        key, val = line.split(":")
        key, val = key.strip(), val.strip()
        model_details[key] = val
        print(f"{key:22s}: {val}")
        print(81 * "-")

HIDDEN_SIZE = int(model_details["hidden_size"])
ENCODER_N_LAYERS = int(model_details["encoder_n_layers"])
DECODER_N_LAYERS = int(model_details["decoder_n_layers"])
MAX_LENGTH = int(model_details["max_length"])
RNN_TYPE = model_details["rnn_type"]
DROPOUT = float(model_details["dropout"])

# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token


def indexesFromSentence(voc, sentence):
    return [voc.word2index[word] for word in sentence.split(" ")] + [EOS_token]


# Turn a Unicode string to plain ASCII, thanks to
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


class Voc:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3  # Count SOS, EOS, PAD


class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0, rnn_type=RNN_TYPE):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding

        # Initialize GRU; the input_size and hidden_size params are both set to 'hidden_size'
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
        outputs = outputs[:, :, : self.hidden_size] + outputs[:, :, self.hidden_size :]
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
        self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1, rnn_type=RNN_TYPE
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


class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, input_length, max_length):
        # Forward input through encoder model
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        if RNN_TYPE == "LSTM":
            hn = encoder_hidden[0][: decoder.n_layers]
            cn = encoder_hidden[1][: decoder.n_layers]
            decoder_hidden = (hn, cn)
        elif RNN_TYPE == "GRU":
            decoder_hidden = encoder_hidden[: decoder.n_layers]
        # Initialize decoder input with SOS_token
        decoder_input = torch.ones(1, 1, device=DEVICE, dtype=torch.long) * SOS_token
        # Initialize tensors to append decoded words to
        all_tokens = torch.zeros([0], device=DEVICE, dtype=torch.long)
        all_scores = torch.zeros([0], device=DEVICE)
        # Iteratively decode one word token at a time
        for _ in range(max_length):
            # Forward pass through decoder
            decoder_output, decoder_hidden = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # Obtain most likely word token and its softmax score
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            # Record token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        # Return collections of word tokens and scores
        return all_tokens, all_scores


def evaluate(
    encoder, decoder, searcher, voc, sentence, max_length
):
    ### Format input sentence as a batch
    # words -> indexes
    indexes_batch = [indexesFromSentence(voc, sentence)]
    # Create lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # Transpose dimensions of batch to match models' expectations
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    # Use appropriate device
    input_batch = input_batch.to(DEVICE)
    lengths = lengths.to("cpu")
    # Decode sentence with searcher
    tokens, scores = searcher(input_batch, lengths, max_length)
    # indexes -> words
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words


def evaluateInput(encoder, decoder, searcher, voc):
    input_sentence = "how much is tablespoon of water?"
    while 1:
        try:
            # Get input sentence
            input_sentence = input("> ")
            # Check if it is quit case
            if input_sentence == "q" or input_sentence == "quit":
                break
            # Normalize sentence
            input_sentence = normalizeString(input_sentence)
            # Evaluate sentence
            output_words = evaluate(encoder, decoder, searcher, voc, input_sentence, MAX_LENGTH)
            # Format and print response sentence
            output_words[:] = [
                x for x in output_words if not (x == "EOS" or x == "PAD")
            ]
            print("Bot:", " ".join(output_words))

        except KeyError:
            print("Error: Encountered unknown word.")


# print("Building encoder and decoder ...")
# Initialize word embeddings
voc = Voc("eval")
print("\nLoading vocabulary")
voc.__dict__ = checkpoint["voc_dict"]
embedding = nn.Embedding(voc.num_words, HIDDEN_SIZE).to(DEVICE)

# Initialize encoder & decoder models
encoder = EncoderRNN(
    HIDDEN_SIZE,
    embedding,
    ENCODER_N_LAYERS,
    DROPOUT,
    RNN_TYPE
).to(DEVICE)
decoder = LuongAttnDecoderRNN(
    model_details["attn_model"],
    embedding,
    HIDDEN_SIZE,
    voc.num_words,
    DECODER_N_LAYERS, 
    DROPOUT,
    model_details["rnn_type"]
).to(DEVICE)

print(
    "Loading embding state dict: "
    + str(embedding.load_state_dict(checkpoint["embedding"]))
)
print("Loading encoder state dict: " + str(encoder.load_state_dict(checkpoint["en"])))
print("Loading decoder state dict: " + str(decoder.load_state_dict(checkpoint["de"])))

# Set dropout layers to eval mode
embedding.eval()
encoder.eval()
decoder.eval()

# Initialize search module
searcher = GreedySearchDecoder(encoder, decoder)

print()
# Begin chatting (uncomment and run the following line to begin)
evaluateInput(encoder, decoder, searcher, voc)

"""
[['where is the world cup in', 'it took place in south africa from june to july .'], 
['when is the next national election ?', 'the united states presidential election of was the th quadrennial presidential election .']]
"""
