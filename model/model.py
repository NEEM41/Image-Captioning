# Building blocks taken from https://github.com/karpathy/nanoGPT/blob/master/model.py
"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional, Tuple


@dataclass
class LMConfig:
    """
    Hyperparameter dataclass for the CLIP-conditioned GPT decoder, 
    covering vocabulary size, context length, transformer depth/width, 
    dropout, and CLIP prefix configuration.
    """
    vocab_size: int = 8195 
    block_size: int = 1024          # caption length (text tokens only)
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.1

    # CLIP conditioning
    clip_dim: int = 768           # ViT-B/32 pooled dim
    prefix_len: int = 8          # number of pseudo tokens
    bias: bool = False
    pad_token: int = 8192


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        """
        Multi-head causal self-attention block with optional Flash Attention acceleration, 
        handling projection, masking, and residual dropout.
        """
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        """
        Compute masked self-attention over the sequence.

        Args:
            x: Tensor of shape (batch, seq_len, n_embd) containing the block’s normalized hidden states.

        Returns:
            Tensor of identical shape after applying causal multi-head self-attention and output projection.
        """

        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, config):
        """Two-layer feed-forward network (GELU + projection) used inside each transformer block."""
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        """
        Apply the feed-forward sublayer.

        Args:
            x: Tensor of shape (batch, seq_len, n_embd) from the attention sublayer.

        Returns:
            Tensor of the same shape after the GELU-activated expansion, projection, and dropout.
        """
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        """Residual transformer block composed of LayerNorm, causal self-attention, and MLP sublayers."""
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        """
        Run a transformer block step.
        Args:
            x: Tensor of shape (batch, seq_len, n_embd) representing the current hidden states.

        Returns:
            Tensor of the same shape after attention and MLP residual updates.
        """
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class PositionalEmbedding(nn.Module):
    def __init__(self, dim: int, max_len: int = 1024):
        """Sinusoidal positional embedding buffer that provides deterministic position encodings 
        up to the configured maximum length."""
        super().__init__()
        pe = torch.zeros(max_len, dim)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float)
            * (-math.log(10000.0) / dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, dim)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Slice positional encodings for the requested length.
        Args:
            x: Tensor whose second dimension (seq_len) determines how many positions to return.

        Returns:
            Tensor of shape (1, seq_len, dim) containing the positional embeddings to add to token features.
        """
        return self.pe[:, : x.size(1)]

class ClipLM(nn.Module):
    def __init__(self, cfg: LMConfig):
        """CLIP-conditioned decoder-only language model that prepends projected CLIP 
        features as learned prefix tokens before autoregressive caption generation."""
        super().__init__()
        self.cfg = cfg
        self.max_seq = cfg.block_size

        self.wte = nn.Embedding(self.cfg.vocab_size, self.cfg.n_embd)
        self.drop = nn.Dropout(self.cfg.dropout)

        # Decoder Blocks
        self.blocks = nn.ModuleList(
            [Block(cfg) for _ in range(cfg.n_layer)]
        )
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.pos_emb = PositionalEmbedding(dim=cfg.n_embd)

        # language modeling head
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        # CLIP -> prefix projector: (B, clip_dim) -> (B, K*n_embd) -> (B, K, n_embd)
        self.clip_proj = nn.Sequential(
            nn.Linear(cfg.clip_dim, 2 * cfg.n_embd),
            nn.GELU(),
            nn.Linear(2 * cfg.n_embd, cfg.prefix_len * cfg.n_embd),
        )

        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * cfg.n_layer))

        # weight tying -> Ususally done in models like GPT2
        self.lm_head.weight = self.wte.weight

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        Original implementation removes transformer.wpe but we switched to 
        sinusodial positional encoding so replaced that now.
        """
        n_params = sum(p.numel() for p in self.parameters())
        # if non_embedding:
        #     n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        """Initialize Linear and Embedding modules with the GPT-2 normal 
        distribution and zero biases to match the original training recipe."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,              # (B, T) text tokens
        clip_emb: torch.Tensor,               # (B, clip_dim)
        labels: Optional[torch.Tensor] = None # (B, T) text tokens (same as input_ids usually)
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Run a forward pass through the captioning model.

        Args:
            input_ids: LongTensor of shape (batch, text_len) with tokenized captions.
            clip_emb: FloatTensor of shape (batch, clip_dim) containing pooled CLIP image embeddings.
            labels: Optional LongTensor matching input_ids for teacher forcing loss computation.

        Returns:
            A tuple of (logits, loss) where logits has shape (batch, prefix_len + text_len, vocab_size) and 
            loss is cross-entropy over next-token predictions when labels are provided.
        """
        B, T = input_ids.shape
        assert T <= self.cfg.block_size, f"T={T} > block_size={self.cfg.block_size}"

        # build prefix embeddings
        prefix = self.clip_proj(clip_emb).view(B, self.cfg.prefix_len, self.cfg.n_embd)  # (B,K,C)

        # text token embeddings
        tok = self.wte(input_ids)  # (B,T,C)

        # concat: (B, K+T, C)
        x = torch.cat([prefix, tok], dim=1)
        # adding positional encoding
        x = self.drop(x + self.pos_emb(x))
        TT = x.size(1)

        # transformer
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)

        logits = self.lm_head(x)  # (B, K+T, vocab)

        loss = None
        if labels is not None:
            # Remove the prefix tokens
            text_logits = logits[:, self.cfg.prefix_len:, :]  # (B,T,V)

            # input[:, :-1], labels[:, 1:] -> next token teacher forcing.
            loss = F.cross_entropy(
                text_logits[:, :-1, :].contiguous().view(-1, self.cfg.vocab_size),
                labels[:, 1:].contiguous().view(-1),
                ignore_index=-100,
                reduction='mean'
            )

        breakpoint()

        return logits, loss

if __name__ == '__main__':
    from dataset.dataset import ClipCaptionDataModule

    dm = ClipCaptionDataModule(
        train_csv="/Users/swornimchhetri/Desktop/all_codes/github_stuff/Image-Captioning/csvs/coco_train_tok.csv",
        val_csv="/Users/swornimchhetri/Desktop/all_codes/github_stuff/Image-Captioning/csvs/coco_val_tok.csv",
        embeddings_root="/Users/swornimchhetri/Desktop/all_codes/github_stuff/Image-Captioning/embeddings/pooled_clip_output",
        pad_id=8192,          # set to your tokenizer pad id
        batch_size=64,
        max_len=69,
        num_workers=0,     # start with 0 on Mac
    )

    dm.setup()

    train_loader = dm.train_dataloader()

    model = ClipLM(LMConfig())
    device = 'mps'
    model = model.to(device)

    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)   # must be on mps
        clip_emb  = batch["clip_emb"].to(device)    # must be on mps
        labels    = batch["labels"].to(device)      # must be on mps
        output = model(input_ids=input_ids,
                       clip_emb=clip_emb,
                       labels=labels)
        breakpoint()