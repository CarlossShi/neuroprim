import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


class Transformer_encoder_net(nn.Module):
    """
    Encoder network based on self-attention transformer
    Inputs :
      h of size      (bsz, nb_nodes+1, dim_emb)    batch of input cities
    Outputs :
      h of size      (bsz, nb_nodes+1, dim_emb)    batch of encoded cities
      score of size  (bsz, nb_nodes+1, nb_nodes+1) batch of attention scores
    """

    def __init__(self, nb_layers, dim_emb, nb_heads, dim_ff, batchnorm):
        super(Transformer_encoder_net, self).__init__()
        assert dim_emb == nb_heads * (dim_emb // nb_heads)  # check if dim_emb is divisible by nb_heads
        self.MHA_layers = nn.ModuleList([nn.MultiheadAttention(dim_emb, nb_heads) for _ in range(nb_layers)])
        self.linear1_layers = nn.ModuleList([nn.Linear(dim_emb, dim_ff) for _ in range(nb_layers)])
        self.linear2_layers = nn.ModuleList([nn.Linear(dim_ff, dim_emb) for _ in range(nb_layers)])
        if batchnorm:
            self.norm1_layers = nn.ModuleList([nn.BatchNorm1d(dim_emb) for _ in range(nb_layers)])
            self.norm2_layers = nn.ModuleList([nn.BatchNorm1d(dim_emb) for _ in range(nb_layers)])
        else:
            self.norm1_layers = nn.ModuleList([nn.LayerNorm(dim_emb) for _ in range(nb_layers)])
            self.norm2_layers = nn.ModuleList([nn.LayerNorm(dim_emb) for _ in range(nb_layers)])
        self.nb_layers = nb_layers
        self.nb_heads = nb_heads
        self.batchnorm = batchnorm

    def forward(self, h):
        # PyTorch nn.MultiheadAttention requires input size (seq_len, bsz, dim_emb)
        h = h.transpose(0, 1)  # size(h)=(nb_nodes, bsz, dim_emb)
        # L layers
        for i in range(self.nb_layers):
            h_rc = h  # residual connection, size(h_rc)=(nb_nodes, bsz, dim_emb)
            h, score = self.MHA_layers[i](h, h,
                                          h)  # size(h)=(nb_nodes, bsz, dim_emb), size(score)=(bsz, nb_nodes, nb_nodes)
            # add residual connection
            h = h_rc + h  # size(h)=(nb_nodes, bsz, dim_emb)
            if self.batchnorm:
                # Pytorch nn.BatchNorm1d requires input size (bsz, dim, seq_len)
                h = h.permute(1, 2, 0).contiguous()  # size(h)=(bsz, dim_emb, nb_nodes)
                h = self.norm1_layers[i](h)  # size(h)=(bsz, dim_emb, nb_nodes)
                h = h.permute(2, 0, 1).contiguous()  # size(h)=(nb_nodes, bsz, dim_emb)
            else:
                h = self.norm1_layers[i](h)  # size(h)=(nb_nodes, bsz, dim_emb)
            # feedforward
            h_rc = h  # residual connection
            h = self.linear2_layers[i](torch.relu(self.linear1_layers[i](h)))
            h = h_rc + h  # size(h)=(nb_nodes, bsz, dim_emb)
            if self.batchnorm:
                h = h.permute(1, 2, 0).contiguous()  # size(h)=(bsz, dim_emb, nb_nodes)
                h = self.norm2_layers[i](h)  # size(h)=(bsz, dim_emb, nb_nodes)
                h = h.permute(2, 0, 1).contiguous()  # size(h)=(nb_nodes, bsz, dim_emb)
            else:
                h = self.norm2_layers[i](h)  # size(h)=(nb_nodes, bsz, dim_emb)
        # Transpose h
        h = h.transpose(0, 1)  # size(h)=(bsz, nb_nodes, dim_emb)
        return h, score


def myMHA(Q, K, V, nb_heads, mask=None, clip_value=None):
    """
    Compute multi-head attention (MHA) given a query Q, key K, value V and attention mask :
      h = Concat_{k=1}^nb_heads softmax(Q_k^T.K_k).V_k
    Note : We did not use nn.MultiheadAttention to avoid re-computing all linear transformations at each call.
    Inputs : Q of size (bsz, dim_emb, 1)                batch of queries
             K of size (bsz, dim_emb, nb_nodes+1)       batch of keys
             V of size (bsz, dim_emb, nb_nodes+1)       batch of values
             mask of size (bsz, nb_nodes+1)             batch of masks of visited cities
             clip_value is a scalar
    Outputs : attn_output of size (bsz, 1, dim_emb)     batch of attention vectors
              attn_weights of size (bsz, 1, nb_nodes+1) batch of attention weights
    """
    device = Q.device
    bsz, nb_nodes, emd_dim = K.size()  # dim_emb must be divisable by nb_heads
    if nb_heads > 1:
        assert V is not None
        # PyTorch view requires contiguous dimensions for correct reshaping
        Q = Q.transpose(1, 2).contiguous().view(bsz * nb_heads, emd_dim // nb_heads, 1).transpose(1, 2).contiguous()  # size(Q)=(bsz*nb_heads, 1, dim_emb//nb_heads)
        K = K.transpose(1, 2).contiguous().view(bsz * nb_heads, emd_dim // nb_heads, nb_nodes).transpose(1, 2).contiguous()  # size(K)=(bsz*nb_heads, nb_nodes+1, dim_emb//nb_heads)
        V = V.transpose(1, 2).contiguous().view(bsz * nb_heads, emd_dim // nb_heads, nb_nodes).transpose(1, 2).contiguous()  # size(V)=(bsz*nb_heads, nb_nodes+1, dim_emb//nb_heads)
    attn_weights = torch.bmm(Q, K.transpose(1, 2)) / Q.size(-1) ** 0.5  # size(attn_weights)=(bsz*nb_heads, 1, nb_nodes+1)
    if clip_value is not None:
        attn_weights = clip_value * torch.tanh(attn_weights)
    if mask is not None:
        if nb_heads > 1:
            mask = torch.repeat_interleave(mask, repeats=nb_heads, dim=0)  # size(mask)=(bsz*nb_heads, nb_nodes+1)
        # attn_weights = attn_weights.masked_fill(mask.unsqueeze(1), float('-inf')) # size(attn_weights)=(bsz*nb_heads, 1, nb_nodes+1)
        if Q.dtype == torch.float32:
            min_float = float('-1e9')
        elif Q.dtype == torch.float16:
            min_float = float(-65504)
        else:
            assert False, 'unknown dtype: {}'.format(Q.dtype)
        attn_weights = attn_weights.masked_fill(mask.unsqueeze(1), min_float)  # size(attn_weights)=(bsz*nb_heads, 1, nb_nodes+1)
    attn_weights = torch.softmax(attn_weights, dim=-1)  # size(attn_weights)=(bsz*nb_heads, 1, nb_nodes+1)
    if V is not None:
        attn_output = torch.bmm(attn_weights, V)  # size(attn_output)=(bsz*nb_heads, 1, dim_emb//nb_heads)
    else:
        attn_output = None
    if nb_heads > 1:
        attn_output = attn_output.transpose(1, 2).contiguous()  # size(attn_output)=(bsz*nb_heads, dim_emb//nb_heads, 1)
        attn_output = attn_output.view(bsz, emd_dim, 1)  # size(attn_output)=(bsz, dim_emb, 1)
        attn_output = attn_output.transpose(1, 2).contiguous()  # size(attn_output)=(bsz, 1, dim_emb)
        attn_weights = attn_weights.view(bsz, nb_heads, 1, nb_nodes)  # size(attn_weights)=(bsz, nb_heads, 1, nb_nodes+1)
        attn_weights = attn_weights.mean(dim=1)  # mean over the heads, size(attn_weights)=(bsz, 1, nb_nodes+1)
    return attn_output, attn_weights


class AutoRegressiveDecoderLayer(nn.Module):
    """
    Single decoder layer based on self-attention and query-attention
    Inputs :
      h_t of size      (bsz, 1, dim_emb)          batch of input queries
      K_att of size    (bsz, nb_nodes+1, dim_emb) batch of query-attention keys
      V_att of size    (bsz, nb_nodes+1, dim_emb) batch of query-attention values
      mask of size     (bsz, nb_nodes+1)          batch of masks of visited cities
    Output :
      h_t of size (bsz, nb_nodes+1)               batch of transformed queries
    """

    def __init__(self, dim_emb, nb_heads, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(AutoRegressiveDecoderLayer, self).__init__()
        self.dim_emb = dim_emb
        self.nb_heads = nb_heads
        self.Wq_selfatt = nn.Linear(dim_emb, dim_emb, **factory_kwargs)
        self.Wk_selfatt = nn.Linear(dim_emb, dim_emb, **factory_kwargs)
        self.Wv_selfatt = nn.Linear(dim_emb, dim_emb, **factory_kwargs)
        self.W0_selfatt = nn.Linear(dim_emb, dim_emb, **factory_kwargs)
        self.W0_att = nn.Linear(dim_emb, dim_emb, **factory_kwargs)
        self.Wq_att = nn.Linear(dim_emb, dim_emb, **factory_kwargs)
        self.W1_MLP = nn.Linear(dim_emb, dim_emb, **factory_kwargs)
        self.W2_MLP = nn.Linear(dim_emb, dim_emb, **factory_kwargs)
        self.BN_selfatt = nn.LayerNorm(dim_emb, **factory_kwargs)
        self.BN_att = nn.LayerNorm(dim_emb, **factory_kwargs)
        self.BN_MLP = nn.LayerNorm(dim_emb, **factory_kwargs)
        self.K_sa = None
        self.V_sa = None

    def reset_selfatt_keys_values(self):
        self.K_sa = None
        self.V_sa = None

    def forward(self, h_t, K_att, V_att, mask):
        bsz = h_t.size(0)
        h_t = h_t.view(bsz, 1, self.dim_emb)  # size(h_t)=(bsz, 1, dim_emb)
        # embed the query for self-attention
        q_sa = self.Wq_selfatt(h_t)  # size(q_sa)=(bsz, 1, dim_emb)
        k_sa = self.Wk_selfatt(h_t)  # size(k_sa)=(bsz, 1, dim_emb)
        v_sa = self.Wv_selfatt(h_t)  # size(v_sa)=(bsz, 1, dim_emb)
        # concatenate the new self-attention key and value to the previous keys and values
        if self.K_sa is None:
            self.K_sa = k_sa  # size(self.K_sa)=(bsz, 1, dim_emb)
            self.V_sa = v_sa  # size(self.V_sa)=(bsz, 1, dim_emb)
        else:
            self.K_sa = torch.cat([self.K_sa, k_sa], dim=1)
            self.V_sa = torch.cat([self.V_sa, v_sa], dim=1)
        # compute self-attention between nodes in the partial tour
        h_t = h_t + self.W0_selfatt(myMHA(q_sa, self.K_sa, self.V_sa, self.nb_heads)[0])  # size(h_t)=(bsz, 1, dim_emb)
        h_t = self.BN_selfatt(h_t.squeeze())  # size(h_t)=(bsz, dim_emb)
        h_t = h_t.view(bsz, 1, self.dim_emb)  # size(h_t)=(bsz, 1, dim_emb)
        # compute attention between self-attention nodes and encoding nodes in the partial tour (translation process)
        q_a = self.Wq_att(h_t)  # size(q_a)=(bsz, 1, dim_emb)
        h_t = h_t + self.W0_att(myMHA(q_a, K_att, V_att, self.nb_heads, mask)[0])  # size(h_t)=(bsz, 1, dim_emb)
        h_t = self.BN_att(h_t.squeeze())  # size(h_t)=(bsz, dim_emb)
        h_t = h_t.view(bsz, 1, self.dim_emb)  # size(h_t)=(bsz, 1, dim_emb)
        # MLP
        h_t = h_t + self.W2_MLP(torch.relu(self.W1_MLP(h_t)))
        h_t = self.BN_MLP(h_t.squeeze(1))  # size(h_t)=(bsz, dim_emb)
        return h_t


class Transformer_decoder_net(nn.Module):
    """
    Decoder network based on self-attention and query-attention transformers
    Inputs :
      h_t of size      (bsz, 1, dim_emb)                            batch of input queries
      K_att of size    (bsz, nb_nodes+1, dim_emb*nb_layers_decoder) batch of query-attention keys for all decoding layers
      V_att of size    (bsz, nb_nodes+1, dim_emb*nb_layers_decoder) batch of query-attention values for all decoding layers
      mask of size     (bsz, nb_nodes+1)                            batch of masks of visited cities
    Output :
      prob_next_node of size (bsz, nb_nodes+1)                      batch of probabilities of next node
    """

    def __init__(self, dim_emb, nb_heads, nb_layers_decoder, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Transformer_decoder_net, self).__init__()
        self.dim_emb = dim_emb
        self.nb_heads = nb_heads
        self.nb_layers_decoder = nb_layers_decoder
        self.decoder_layers = nn.ModuleList(
            [AutoRegressiveDecoderLayer(dim_emb, nb_heads, **factory_kwargs) for _ in range(nb_layers_decoder - 1)])
        self.Wq_final = nn.Linear(dim_emb, dim_emb, **factory_kwargs)

    # Reset to None self-attention keys and values when decoding starts
    def reset_selfatt_keys_values(self):
        for l in range(self.nb_layers_decoder - 1):
            self.decoder_layers[l].reset_selfatt_keys_values()

    def forward(self, h_t, K_att, V_att, mask):
        for l in range(self.nb_layers_decoder):
            K_att_l = K_att[:, :, l * self.dim_emb:(l + 1) * self.dim_emb]  # size(K_att_l)=(bsz, nb_nodes+1, dim_emb)
            if l < self.nb_layers_decoder - 1:  # decoder layers with multiple heads (intermediate layers)
                V_att_l = V_att[:, :, l * self.dim_emb:(l + 1) * self.dim_emb]  # size(V_att_l)=(bsz, nb_nodes+1, dim_emb)
                h_t = self.decoder_layers[l](h_t, K_att_l, V_att_l, mask)  # (eq16)
            else:  # decoder layers with single head (final layer)
                q_final = self.Wq_final(h_t)  # (eq22)
                bsz = h_t.size(0)
                q_final = q_final.view(bsz, 1, self.dim_emb)
                attn_weights = myMHA(q_final, K_att_l, None, 1, mask, 10)[1]  # (eq21)
        prob_next_node = attn_weights.squeeze(1)
        return prob_next_node


def generate_positional_encoding(d_model, max_len):
    """
    Create standard transformer PEs.
    Inputs :
      d_model is a scalar correspoding to the hidden dimension
      max_len is the maximum length of the sequence
    Output :
      pe of size (max_len, d_model), where d_model=dim_emb, max_len=1000
    """
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe