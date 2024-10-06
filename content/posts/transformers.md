+++
title = "Implementing the Transformer Model from Scratch "
description = "A super comprehensive and detailed guide to understanding the Transformer model and implementing it using PyTorch."
date = 2024-10-23

[taxonomies]
tags = ["ML Research Paper Implementation"]
+++

### Introduction

The Transformer, with its parallel processing capabilities, allowed for more efficient and scalable models, making it easier to train them on large datasets. It also demonstrated superior performance in several NLP tasks, such as sentiment analysis and text generation tasks. The Transformer model has since become the foundation for many state-of-the-art NLP models, such as BERT, GPT-2, and T5.

### Main Components of the Transformer Architecture

![Transformer Architecture](https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/transformers-dark.svg)

The Transformer Architecture Consists of 2 main Components ->

- Encoder -  The encoder receives an input and builds an Embedding of it's Features. This means that the model learns to understand the association of words or sequence.

- Decoder - The decoder uses the encoder’s output embeddings along with other inputs to generate a target sequence.

### Input Embeddings

Whenever we use a dataset , and try to Train a Model on it , we always convert explicitly or implicitly to a representation which the model can interpret / understand and then reconvert it into a representation we understand , the Function of Input Embedding Block in the Transformer Architecture is just that only. In the Orignal Paper the Authors used , the Input Block with an Embedding Dimension of 512.To prevent the input Embeddings from being extremely small , we normalize them by Multiplying the by root of EmbeddingDimension

#### Implementation of the Input Embedding Block

```python
    import torch
    import torch.nn as nn
    import math
        
    class InputEmbeddingBlock(nn.Module):

        def __init__(self,embedding_dim:int,vocab_size:int):
                
            super().__init__()
            self.embedding_dim = embedding_dim # Reffered in the paper as d_model, (Size == 512)
            self.vocab_size = vocab_size ## Size of the Vocabulary of the input 
            self.embedding = nn.Embedding(self.vocab_size,self.embedding_dim)

        def forward(self,x):
            return self.embedding_dim(x)*math.sqrt(self.embedding_dim) ## This is done to help Prevent the Size of Input Embedding being diminished

```

### Positional Encoding

The input now is converted into input Embeddings of Dimension 512 , but unless we don't provide a signal for the encoder on the relative or absolute position of the tokens in the sequence the Model can't learn the corresponding association
to get around that probelm the authors have provided a positional Encoding for a token based on if its index is an even number or an odd number , these encodings are computed only once and in the paper are not learned by the model.

#### Implementation of the Positional Encoding Block

```python
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """
    Positional Encoding module for Transformer models.

    Args:
        embedding_dim (int): The dimension of the input embeddings.
        sequence_len (int): The length of the input sequence.
        dropout (float): The dropout probability.
    """
    def __init__(self, embedding_dim: int, sequence_len: int, dropout: float):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.sequence_len = sequence_len
        self.dropout = nn.Dropout(dropout)

        # Creating a matrix of size (sequence_len,embedding_dim)
        positional_encoding = torch.zeros(sequence_len, self.embedding_dim)

        # Create a vector of shape (sequence_len,1)
        position = torch.arange(0, sequence_len, dtype=torch.float).unsqueeze(dim=1)
        division_term = torch.exp(torch.arange(0, embedding_dim, 2)).float() * (-torch.log(10000.0) / embedding_dim)

        # Apply the sin formula to the even positions and cosine formula to the odd positions
        positional_encoding[:, 0::2] = torch.sin(position * division_term) # Every two Terms even -> 0 -> 2 -> 4 
        positional_encoding[:, 1::2] = torch.cos(position * division_term) # Every two Terms odd -> 1 -> 3 -> 5
        
        positional_encoding = positional_encoding.unsqueeze(dim=0)
        self.register_buffer('positional_encoding',positional_encoding)
        
    def forward(self,x):
        x = x + (self.pe[:,:x.shape[1],:]).requires_grad(False)
        return self.dropout(x)

```

### Layer Normalization

These are the Add and Norm Layer in the Architecture these help scaling input tensor with Layer the LayerNormalization Block is already implemented in Pytorch

```python

import torch
import torch.nn as nn


class LayerNormalization(nn.Module):
    """
    Applies layer normalization to the input tensor.

    Args:
        eps (float, optional): A value added to the denominator for numerical stability. Default is 1e-5.
    """

    def __init__(self, eps: float = 1e-5) -> None:
        self.eps = eps

    def forward(self, x):
        """
        Applies layer normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.
        """
        return nn.LayerNorm(x, eps=self.eps)


```

### Simple MLP or FeedForward Block

Really Simple MLP consisting of 2 Linear Layers with the ReLU activation function b/w them also using Dropout Layer for overfitting prevention.

### Implemention of the FeedForward Block

```python
import torch.nn as nn

class FeedForwardBlock(nn.Module):
    """
    A feed-forward block in the Transformer model.

    Args:
        embedding_dim (int): The dimensionality of the input embeddings.
        feed_forward_dim (int): The dimensionality of the hidden layer in the feed-forward network.
        dropout (float): The dropout probability.

    Attributes:
        linear_1 (nn.Linear): The first linear layer.
        dropout (nn.Dropout): The dropout layer.
        linear_2 (nn.Linear): The second linear layer.
    """

    def __init__(self, embedding_dim, feed_forward_dim, dropout) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(embedding_dim, feed_forward_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(feed_forward_dim, embedding_dim)
        
    def forward(self, x):
        """
        Forward pass of the feed-forward block.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        x = self.linear_1(x)
        x = nn.ReLU(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        return x 
```

### Multihead Attention Block

The Multi-Head Attention block receives the input data split into queries, keys, and values organized into matrices 𝑄, 𝐾, and 𝑉. Each matrix contains different facets of the input, and they have the same dimensions as the input.We then linearly transform each matrix by their respective weight matrices 𝑊^Q, 𝑊^K, and 𝑊^V. These transformations will result in new matrices 𝑄′, 𝐾′, and 𝑉′, which will be split into smaller matrices corresponding to different heads ℎ, allowing the model to attend to information from different representation subspaces in parallel. This split creates multiple sets of queries, keys, and values for each head. Finally, we concatenate every head into an 𝐻 matrix, which is then transformed by another weight matrix 𝑊𝑜 to produce the multi-head attention output, a matrix 𝑀𝐻−𝐴 that retains the input dimensionality.

### Implementation of Multihead Attention Block

```python


import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        """
        Initializes the MultiHeadAttention module.

        Args:
            embedding_dim (int): The input and output dimension of the model.
            num_heads (int): The number of attention heads.

        Raises:
            AssertionError: If embedding_dim is not divisible by num_heads.
        """
        super(MultiHeadAttention, self).__init__()
        assert embedding_dim % num_heads == 0, "embedding_dim must be divisible by num_heads"
        
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.d_k = self.embedding_dim // num_heads
        
        self.W_q = nn.Linear(embedding_dim,embedding_dim)
        self.W_k = nn.Linear(embedding_dim, embedding_dim)
        self.W_v = nn.Linear(embedding_dim, embedding_dim)
        self.W_o = nn.Linear(embedding_dim, embedding_dim)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        Performs scaled dot product attention.

        Args:
            Q (torch.Tensor): The query tensor of shape (batch_size, seq_length, embedding_dim).
            K (torch.Tensor): The key tensor of shape (batch_size, seq_length, embedding_dim).
            V (torch.Tensor): The value tensor of shape (batch_size, seq_length, embedding_dim).
            mask (torch.Tensor, optional): The attention mask tensor of shape (batch_size, seq_length, seq_length).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, seq_length, embedding_dim).
        """
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output
        
    def split_heads(self, x):
        """
        Splits the input tensor into multiple heads.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, seq_length, embedding_dim).

        Returns:
            torch.Tensor: The tensor with shape (batch_size, num_heads, seq_length, d_k).
        """
        batch_size, seq_length, embedding_dim = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        """
        Combines the heads of the input tensor.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, num_heads, seq_length, d_k).

        Returns:
            torch.Tensor: The tensor with shape (batch_size, seq_length, embedding_dim).
        """
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.embedding_dim)
        
    def forward(self, Q, K, V, mask=None):
        """
        Performs forward pass of the MultiHeadAttention module.

        Args:
            Q (torch.Tensor): The query tensor of shape (batch_size, seq_length, embedding_dim).
            K (torch.Tensor): The key tensor of shape (batch_size, seq_length, embedding_dim).
            V (torch.Tensor): The value tensor of shape (batch_size, seq_length, embedding_dim).
            mask (torch.Tensor, optional): The attention mask tensor of shape (batch_size, seq_length, seq_length).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, seq_length, embedding_dim).
        """
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output

```

### Encoder Block 

An Encoder layer consists of a Multi-Head Attention layer, a Position-wise Feed-Forward layer, and two Layer Normalization layers.
The EncoderLayer class initializes with input parameters and components, including a MultiHeadAttention module, a PositionWiseFeedForward module, two layer normalization modules, and a dropout layer. The forward methods computes the encoder layer output by applying self-attention, adding the attention output to the input tensor, and normalizing the result. Then, it computes the position-wise feed-forward output, combines it with the normalized self-attention output, and normalizes the final result before returning the processed tensor.

### Implementation of the Encoder Block

```python
import torch
import torch.nn as nn 


class EncoderBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, feed_forward_dim, dropout):
        """
        Initializes an EncoderLayer module.

        Args:
            embedding_dim (int): The dimensionality of the input and output feature vectors.
            num_heads (int): The number of attention heads.
            feed_forward_dim (int): The dimensionality of the feed-forward layer.
            dropout (float): The dropout probability.

        """
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(embedding_dim, num_heads)
        self.feed_forward = FeedForward(embedding_dim,feed_forward_dim)
        self.norm1 = LayerNormalization(embedding_dim)
        self.norm2 = LayerNormalization(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        """
        Performs a forward pass of the EncoderLayer module.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, seq_len, d_model).
            mask (torch.Tensor): The attention mask tensor of shape (batch_size, seq_len, seq_len).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, seq_len, d_model).

        """
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x
```

### Decoder Block

After Encoder the Keys and Values from the output are consumed and Query is provided by the Output Embedding in the Decoder the decoder layer consists of two Multi-Head Attention layers, a Position-wise Feed-Forward layer, and three Layer Normalization layers. The forward method computes the decoder layer output by performing the following steps:

- Calculate the masked self-attention output and add it to the input tensor, followed by dropout and layer normalization.
- Compute the cross-attention output between the decoder and encoder outputs, and add it to the normalized masked self-attention output, followed by dropout and layer normalization.
- Calculate the position-wise feed-forward output and combine it with the normalized cross-attention output, followed by dropout and layer normalization.
- Return the processed tensor.
- These operations enable the decoder to generate target sequences based on the input and the encoder output.

### Implementation of the Decoder Block

```python

import torch 
import torch.nn as nn

class DecoderBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, feed_forward_dim, dropout):
        """
        Initializes a DecoderLayer module.

        Args:
            embedding_dim (int): The dimension of the input embeddings.
            num_heads (int): The number of attention heads.
            feed_forward_dim (int): The dimension of the feed-forward layer.
            dropout (float): The dropout probability.

        """
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(embedding_dim, num_heads)
        self.cross_attn = MultiHeadAttention(embedding_dim, num_heads)
        self.feed_forward = FeedForwardBlock(embedding_dim, feed_forward_dim)
        self.norm1 = LayerNormalization(d_model)
        self.norm2 = LayerNormalization(d_model)
        self.norm3 = LayerNormalization(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask, tgt_mask):
        """
        Performs a forward pass of the DecoderLayer module.

        Args:
            x (torch.Tensor): The input tensor.
            enc_output (torch.Tensor): The output of the encoder.
            src_mask (torch.Tensor): The mask for the source sequence.
            tgt_mask (torch.Tensor): The mask for the target sequence.

        Returns:
            torch.Tensor: The output tensor.

        """
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x
```

### Building the Transformer

Now that we have built every layer Lets combine them to Build the Transformer

```python

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, embedding_dim, num_heads, num_layers, feed_forward_dim, max_seq_length, dropout):
        """
        Initializes the Transformer model.

        Args:
        - src_vocab_size (int): The size of the source vocabulary.
        - tgt_vocab_size (int): The size of the target vocabulary.
        - embedding_dim (int): The dimension of the word embeddings.
        - num_heads (int): The number of attention heads.
        - num_layers (int): The number of encoder and decoder layers.
        - feed_forward_dim (int): The dimension of the feed-forward layer.
        - max_seq_length (int): The maximum sequence length.
        - dropout (float): The dropout rate.

        """
        super(Transformer, self).__init__()
        self.encoder_embedding = InputEmbeddingBlock(src_vocab_size, embedding_dim)
        self.decoder_embedding = InputEmbeddingBlock(tgt_vocab_size, embedding_dim)
        self.positional_encoding = PositionalEncoding(embedding_dim, max_seq_length)
        self.encoder_layers = nn.ModuleList([EncoderBlock(embedding_dim, num_heads, feed_forward_dim, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderBlock(embedding_dim, num_heads, feed_forward_dim, dropout) for _ in range(num_layers)])
        self.fc = nn.Linear(embedding_dim, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        """
        Generates masks for the source and target sequences.

        Args:
        - src (Tensor): The source sequence tensor.
        - tgt (Tensor): The target sequence tensor.

        Returns:
        - src_mask (Tensor): The source mask tensor.
        - tgt_mask (Tensor): The target mask tensor.

        """
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        """
        Performs forward pass of the Transformer model.

        Args:
        - src (Tensor): The source sequence tensor.
        - tgt (Tensor): The target sequence tensor.

        Returns:
        - output (Tensor): The output tensor.

        """
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output)
        return output
```