# models/rnn.py

import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Recurrent Neural Network (RNN) language model using a vanilla RNN cell.
"""

class RNNCell(nn.Module):
    """
    The RNNCell processes one time step of the input and updates the hidden state.
    It takes the input at the current time step x_{t} and the hidden state from the
    previous time step h_{t-1} and computes the new hidden state h_{t}.
    """
    def __init__(self, config):
        super().__init__()
        self.xh_to_h = nn.Linear(config.n_embd + config.n_embd2, config.n_embd2)

    def forward(self, xt, hprev):
        xh = torch.cat([xt, hprev], dim=1)
        ht = torch.tanh(self.xh_to_h(xh))
        return ht

class RNN(nn.Module):
    """
    The RNN model that uses the RNNCell to process sequences.
    """
    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size
        self.vocab_size = config.vocab_size

        config.n_embd2 = config.n_embd2 if config.n_embd2 is not None else config.n_embd

        # Starting hidden state parameter
        self.start = nn.Parameter(torch.zeros(1, config.n_embd2))

        # Token embedding table
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)

        # RNN cell
        self.cell = RNNCell(config)

        # Output layer (language modeling head)
        self.lm_head = nn.Linear(config.n_embd2, self.vocab_size)

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, targets=None):
        """
        Args:
            idx: Input indices tensor of shape (batch_size, sequence_length).
            targets: Target indices tensor of shape (batch_size, sequence_length).
        Returns:
            logits: Logits tensor of shape (batch_size, sequence_length, vocab_size).
            loss: Cross-entropy loss between logits and targets (if targets are provided).
        """
        device = idx.device
        b, t = idx.size()

        # Embed the input indices
        emb = self.wte(idx)  # Shape: (batch_size, sequence_length, n_embd)

        # Initialize the hidden state
        hprev = self.start.expand(b, -1)  # Shape: (batch_size, n_embd2)

        hiddens = []
        # Process the input sequence one time step at a time
        for i in range(t):
            xt = emb[:, i, :]  # Shape: (batch_size, n_embd)
            ht = self.cell(xt, hprev)  # Shape: (batch_size, n_embd2)
            hprev = ht
            hiddens.append(ht)

        # Stack hidden states
        hidden = torch.stack(hiddens, dim=1)  # Shape: (batch_size, sequence_length, n_embd2)

        # Compute logits
        logits = self.lm_head(hidden)  # Shape: (batch_size, sequence_length, vocab_size)

        # Compute loss if targets are provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )

        return logits, loss
