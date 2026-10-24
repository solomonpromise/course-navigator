"""Small models whose tensor shapes and learning mechanisms can be inspected.

All scratch models return logits of shape (batch, topics). Cross-entropy expects
these raw scores; softmax belongs in prediction, not before CrossEntropyLoss.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from .data import MAX_LENGTH, LABELS


class RecurrentClassifier(nn.Module):
    """Learn word embeddings, read a sequence and classify its final state.

    Args:
        vocabulary_size: Number of train-fitted word IDs, including PAD/UNK.
        kind: RNN, LSTM or GRU; the rest of the pipeline stays identical.
        width: Embedding and hidden-state dimensions in this classroom model.

    Packing removes padding steps. Without it, a 'final state' could describe
    several padding tokens rather than the end of the learner's sentence.
    """
    def __init__(self, vocabulary_size: int, kind: str = "RNN", width: int = 64):
        super().__init__()
        if kind not in {"RNN", "LSTM", "GRU"}:
            raise ValueError("Choose RNN, LSTM or GRU")
        self.kind = kind
        self.embedding = nn.Embedding(vocabulary_size, width, padding_idx=0)
        self.recurrent = getattr(nn, kind)(width, width, batch_first=True)
        self.dropout = nn.Dropout(0.15)
        self.head = nn.Linear(width, len(LABELS))

    def forward(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(ids)  # (B,T) -> (B,T,D)
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, state = self.recurrent(packed)
        # LSTM returns (hidden state, cell state); classification uses hidden.
        hidden = state[0] if self.kind == "LSTM" else state
        return self.head(self.dropout(hidden[-1]))  # (B,D) -> (B,C)


class SelfAttention(nn.Module):
    """Implement multi-head scaled dot-product attention explicitly.

    This encoder can inspect the entire input question. Padding keys are masked;
    there is no causal mask because we are not predicting the next token.
    """
    def __init__(self, width: int = 64, heads: int = 4):
        super().__init__()
        if width % heads:
            raise ValueError("Embedding width must divide evenly into heads")
        self.heads, self.head_width = heads, width // heads
        self.qkv = nn.Linear(width, 3 * width)
        self.output = nn.Linear(width, width)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, inspect: bool = False):
        batch, length, width = x.shape
        # Separate projections, then (B,T,D) -> (B,H,T,D/H).
        q, k, v = [part.reshape(batch, length, self.heads, self.head_width).transpose(1, 2)
                   for part in self.qkv(x).chunk(3, dim=-1)]
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_width)
        scores = scores.masked_fill(~mask[:, None, None, :], -1e4)
        weights = scores.softmax(dim=-1)  # each query distributes weight across valid keys
        mixed = self.dropout(weights) @ v
        mixed = mixed.transpose(1, 2).reshape(batch, length, width)
        output = self.output(mixed)
        return (output, weights) if inspect else output


class EncoderBlock(nn.Module):
    """Mix tokens with attention, then transform each token with an MLP.

    Pre-normalization and residual additions stabilize the update path. The
    feed-forward network uses the same parameters independently at each token.
    """
    def __init__(self, width: int = 64):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.attention = SelfAttention(width)
        self.ffn = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(),
                                 nn.Dropout(0.1), nn.Linear(4 * width, width))
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask):
        x = x + self.dropout(self.attention(self.norm1(x), mask))
        return x + self.dropout(self.ffn(self.norm2(x)))


class TinyTransformer(nn.Module):
    """A two-block encoder built from the explicit attention implementation."""
    def __init__(self, vocabulary_size: int, width: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, width, padding_idx=0)
        self.position = nn.Embedding(MAX_LENGTH, width)
        self.blocks = nn.ModuleList([EncoderBlock(width) for _ in range(2)])
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, len(LABELS))

    def forward(self, ids, lengths):
        mask = ids.ne(0)
        positions = torch.arange(ids.shape[1], device=ids.device)
        x = self.embedding(ids) + self.position(positions)[None, :, :]
        for block in self.blocks:
            x = block(x, mask)
        # Mean only over real tokens; padding vectors must not dilute the mean.
        pooled = (self.norm(x) * mask[:, :, None]).sum(1) / mask.sum(1, keepdim=True)
        return self.head(pooled)


class PretrainedClassifier(nn.Module):
    """Reuse a pretrained BERT encoder and learn a task-specific topic head.

    The base checkpoint supplies representations, not our six topic labels.
    Both frozen-feature and full fine-tuning experiments share this interface.
    """
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(0.1)
        self.head = nn.Linear(encoder.config.hidden_size, len(LABELS))

    def forward(self, input_ids, attention_mask, token_type_ids):
        states = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                              token_type_ids=token_type_ids).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(states.dtype)
        pooled = (states * mask).sum(1) / mask.sum(1).clamp_min(1)
        return self.head(self.dropout(pooled))
