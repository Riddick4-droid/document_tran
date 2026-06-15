import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    """multi-head attention-scaled dot product attention"""
    def __init__(self, d_model:int, num_heads:int, dropout: float=0.1, bias:bool=True):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be whole number divisible by the num_heads"
        self.d_model = d_model
        self.num_heads =  num_heads
        self.d_k = d_model // num_heads
        self.scale = math.sqrt(self.d_k)

        #define the linear projections
        self.W_q = nn.Linear(d_model, d_model, bias=bias)
        self.W_k = nn.Linear(d_model, d_model, bias=bias)
        self.W_v = nn.Linear(d_model, d_model, bias=bias)
        self.W_o = nn.Linear(d_model, d_model, bias=bias)

        self.dropout = nn.Dropout(dropout)


    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: (B, L_q, D)
            key:   (B, L_k, D)
            value: (B, L_k, D)
            mask:  optional attention mask (B, 1, 1, L_k) or (B, 1, L_q, L_k)
        Returns:
            output: (B, L_q, D)
            attention_weights: (B, num_heads, L_q, L_k)
        """
        B,L_q,_ = query.shape
        _,L_k,_ = key.shape

        Q = self.W_q(query).view(B,L_q, self.num_heads, self.d_k).transpose(1,2)
        K = self.W_k(key).view(B,L_k, self.num_heads, self.d_k).transpose(1,2)
        V = self.W_v(value).view(B,L_k, self.num_heads, self.d_k).transpose(1,2)

        attn_scores = torch.matmul(Q,K.transpose(-2,-1))/self.scale

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask==0, float("-inf"))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights,V) #(B,h,L_q,d_K)
        attn_output = attn_output.transpose(1,2).contiguous().view(B,L_q,self.d_model)
        output = self.W_o(attn_output)

        return output, attn_weights
    