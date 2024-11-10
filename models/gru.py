# models/gru.py

import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Gated Recurrent Unit (GRU) language model.
"""

class GRUCell(nn.Module):
    """
    The GRUCell processes one time step of the input and updates the hidden state.
    It uses gating mechanisms to control the flow of information.
    """
    def __init__(self, config):
        super().__init__()
        input_size = config.n_embd
        hidden_size = config.n_embd2

        # Linear layers for update gate (z), reset gate (r), and candidate hidden state (h~)
        self.xh_to_z = nn.Linear(input_size + hidden_size, hidden_size)
        self.xh_to_r = nn.Linear(input_size + hidden_size, hidden_size)
        self.xh_to_hbar = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, xt, hprev):
        xh = torch.cat([xt, hprev], dim=1)

        # Update gate
        z = torch.sigmoid(self.xh_to_z(xh))

        # Reset gate
        r = torch.sigmoid(self.xh_to_r(xh))

        # Candidate hidden state
        xh_reset = torch.cat([xt, r * hprev], dim=1)
        hbar = torch.tanh(self.xh_to_hbar(xh_reset))

        # New hidden state
        ht = (1 - z) * hprev + z * hbar
        return ht

class GRU(nn.Module):
    """
    The GRU model that uses the GRUCell to process sequences.
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

        # GRU cell
        self.cell = GRUCell(config)

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
