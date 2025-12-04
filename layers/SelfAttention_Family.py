import torch
import torch.nn as nn
import numpy as np
from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask
from reformer_pytorch import LSHSelfAttention
from einops import rearrange, repeat
import torch.nn.functional as F
from math import sqrt
import pennylane as qml
import numpy as np
import math


class DSAttention(nn.Module):
    '''De-stationary Attention'''
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(DSAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        tau = 1.0 if tau is None else tau.unsqueeze(
            1).unsqueeze(1)  # B x 1 x 1 x 1
        delta = 0.0 if delta is None else delta.unsqueeze(
            1).unsqueeze(1)  # B x 1 x 1 x S

        # De-stationary Attention, rescaling pre-softmax score with learned de-stationary factors
        scores = torch.einsum("blhe,bshe->bhls", queries, keys) * tau + delta

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None

class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        # real U = U_part(factor*ln(L_k))*L_q
        index_sample = torch.randint(L_K, (L_Q, sample_k))
        K_sample = K_expand[:, :, torch.arange(
            L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(
            Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H,
                                                L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            # requires that L_Q == L_V, i.e. for self-attention only
            assert (L_Q == L_V)
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) /
                     L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[
                                                  None, :, None], index, :] = attn
            return context_in, attns
        else:
            return context_in, None

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * \
                 np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * \
            np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(
            queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(
            context, values, scores_top, index, L_Q, attn_mask)

        return context.contiguous(), attn


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            #tau=tau,
            #delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class ReformerLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None, causal=False, bucket_size=4, n_hashes=4):
        super().__init__()
        self.bucket_size = bucket_size
        self.attn = LSHSelfAttention(
            dim=d_model,
            heads=n_heads,
            bucket_size=bucket_size,
            n_hashes=n_hashes,
            causal=causal
        )

    def fit_length(self, queries):
        # inside reformer: assert N % (bucket_size * 2) == 0
        B, N, C = queries.shape
        if N % (self.bucket_size * 2) == 0:
            return queries
        else:
            # fill the time series
            fill_len = (self.bucket_size * 2) - (N % (self.bucket_size * 2))
            return torch.cat([queries, torch.zeros([B, fill_len, C]).to(queries.device)], dim=1)

    def forward(self, queries, keys, values, attn_mask, tau, delta):
        # in Reformer: defalut queries=keys
        B, N, C = queries.shape
        queries = self.attn(self.fit_length(queries))[:, :N, :]
        return queries, None


class TwoStageAttentionLayer(nn.Module):
    '''
    The Two Stage Attention (TSA) Layer
    input/output shape: [batch_size, Data_dim(D), Seg_num(L), d_model]
    '''

    def __init__(self, configs,
                 seg_num, factor, d_model, n_heads, d_ff=None, dropout=0.1):
        super(TwoStageAttentionLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.time_attention = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                           output_attention=configs.output_attention), d_model, n_heads)
        self.dim_sender = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                       output_attention=configs.output_attention), d_model, n_heads)
        self.dim_receiver = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                         output_attention=configs.output_attention), d_model, n_heads)
        self.router = nn.Parameter(torch.randn(seg_num, factor, d_model))

        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))
        self.MLP2 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # Cross Time Stage: Directly apply MSA to each dimension
        batch = x.shape[0]
        time_in = rearrange(x, 'b ts_d seg_num d_model -> (b ts_d) seg_num d_model')
        time_enc, attn = self.time_attention(
            time_in, time_in, time_in, attn_mask=None, tau=None, delta=None
        )
        dim_in = time_in + self.dropout(time_enc)
        dim_in = self.norm1(dim_in)
        dim_in = dim_in + self.dropout(self.MLP1(dim_in))
        dim_in = self.norm2(dim_in)

        # Cross Dimension Stage: use a small set of learnable vectors to aggregate and distribute messages to build the D-to-D connection
        dim_send = rearrange(dim_in, '(b ts_d) seg_num d_model -> (b seg_num) ts_d d_model', b=batch)
        batch_router = repeat(self.router, 'seg_num factor d_model -> (repeat seg_num) factor d_model', repeat=batch)
        dim_buffer, attn = self.dim_sender(batch_router, dim_send, dim_send, attn_mask=None, tau=None, delta=None)
        dim_receive, attn = self.dim_receiver(dim_send, dim_buffer, dim_buffer, attn_mask=None, tau=None, delta=None)
        dim_enc = dim_send + self.dropout(dim_receive)
        dim_enc = self.norm3(dim_enc)
        dim_enc = dim_enc + self.dropout(self.MLP2(dim_enc))
        dim_enc = self.norm4(dim_enc)

        final_out = rearrange(dim_enc, '(b seg_num) ts_d d_model -> b ts_d seg_num d_model', b=batch)

        return final_out
        

class TimeAttention(nn.Module):
    def __init__(self, mask_flag=True, scale=None, attention_dropout=0.1, output_attention=False, d_model=512, num_heads=8, max_len=100, covariate=False, flash_attention=False):
        super(TimeAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.covariate = covariate
        self.flash_attention = flash_attention
        self.qk_proj = QueryKeyProjection(dim=d_model, num_heads=num_heads, proj_layer=RotaryProjection, kwargs=dict(max_len=max_len),
                                          partial_factor=(0.0, 0.5),)
        self.attn_bias = BinaryAttentionBias(dim=d_model, num_heads=num_heads)

    def forward(self, queries, keys, values, attn_mask, n_vars, n_tokens, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape

        # [B, H, L, E]
        queries = queries.permute(0, 2, 1, 3)
        keys = keys.permute(0, 2, 1, 3)
        if self.flash_attention:
            values = values.permute(0, 2, 1, 3)

        seq_id = torch.arange(n_tokens * n_vars)
        seq_id = repeat(seq_id, 'n -> b h n', b=B, h=H)

        queries, keys = self.qk_proj(
            queries, keys, query_id=seq_id, kv_id=seq_id)

        scale = self.scale or 1. / sqrt(E)

        var_id = repeat(torch.arange(n_vars),
                        'C -> (C n_tokens)', n_tokens=n_tokens)
        var_id = repeat(var_id, 'L -> b h L', b=B, h=1).to(queries.device)

        attn_bias = self.attn_bias(var_id, var_id)

        if self.mask_flag:
            if attn_mask is None:
                if self.covariate:
                    attn_mask = TimerCovariateMask(
                        B, n_vars, n_tokens, device=queries.device)
                else:
                    attn_mask = TimerMultivariateMask(
                        B, n_vars, n_tokens, device=queries.device)
            attn_mask = attn_bias.masked_fill(attn_mask.mask, float("-inf"))
        else:
            attn_mask = attn_bias

        if self.flash_attention:
            V = torch.nn.functional.scaled_dot_product_attention(
                queries, keys, values, attn_mask)
        else:
            scores = torch.einsum("bhle,bhse->bhls", queries, keys)
            scores += attn_mask
            
            A = self.dropout(torch.softmax(scale * scores, dim=-1))
            V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), None
        else:
            return V.contiguous(), None
            

class QuantumAttentionOld(nn.Module):
    def __init__(self, num_qubits=4, mask_flag=True, scale=None, attention_dropout=0.1,
                 output_attention=False, entanglement_factor=0.5):
        """
        QuantumAttention module with Variational Quantum Eigensolver (VQE) for attention score computation.
        """
        super(QuantumAttentionOld, self).__init__()
        self.num_qubits = num_qubits
        self.mask_flag = mask_flag
        self.scale = scale
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.entanglement_factor = entanglement_factor

        # Initialize quantum device
        self.dev = qml.device("default.qubit", wires=self.num_qubits)

        # Trainable quantum parameters
        self.q_params = nn.Parameter(torch.rand(self.num_qubits))

        # Define QNode once with interface torch
        self.qnode = qml.QNode(self.variational_circuit, self.dev, interface='torch', diff_method="backprop")

    def variational_circuit(self, params):
        """Quantum circuit for attention score calculation using VQE."""
        # Encode trainable parameters
        for i in range(self.num_qubits):
            qml.RY(params[i], wires=i)
       
        # Entangle qubits
        for i in range(self.num_qubits - 1):
            qml.CNOT(wires=[i, i + 1])

        return qml.expval(qml.PauliZ(0))  # Scalar output

    def compute_quantum_attention(self, x):
        """
        Computes quantum attention score.
        Currently uses a single scalar output from the circuit,
        optionally you could expand this to multiple outputs per head.
        """
        # This can be modified to use encoded input `x` if desired
        return self.qnode(self.q_params).float()

    def forward(self, queries, keys, values, attn_mask=None):
        """
        Forward pass for the Quantum Attention using VQE.

        :param queries: (B, L, H, E)
        :param keys:    (B, S, H, D)
        :param values:  (B, S, H, D)
        :param attn_mask: optional attention mask
        :return: (attention output, optional attention weights)
        """
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        # Classical attention (dot product similarity)
        classical_scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        # Quantum attention: compute one score per batch-head (you can expand this)
        quantum_score = self.compute_quantum_attention(classical_scores).to(queries.device)
        quantum_score = quantum_score.expand(B, H, L, S)  # broadcast

        # Entanglement term (optional)
        entanglement_scores = torch.einsum("blhd,bshe->bhls", values, keys)

        # Combine scores
        scores = quantum_score + self.entanglement_factor * entanglement_scores

        # Apply mask
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = torch.triu(torch.ones(L, S), diagonal=1).bool().to(queries.device)
            scores.masked_fill_(attn_mask.unsqueeze(0).unsqueeze(0), -float('inf'))

        # Softmax normalization and dropout
        A = self.dropout(F.softmax(scale * scores, dim=-1))

        # Weighted sum of values
        V = torch.einsum("bhls,bshd->blhd", A, values)

        return (V.contiguous(), A) if self.output_attention else (V.contiguous(), None)

class QuantumVariationalLayer(nn.Module):
    def __init__(self, n_qubits, n_layers):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        self.dev = qml.device("default.qubit", wires=n_qubits)
        
        self.weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        
        self.qnode = qml.QNode(self._circuit, self.dev, interface='torch', diff_method="backprop")
        self.q_layer = qml.qnn.TorchLayer(self.qnode, self.weight_shapes)

    def _circuit(self, inputs, weights):
        for i in range(self.n_qubits):
            qml.RX(inputs[..., i] * np.pi, wires=i)
        for l in range(self.n_layers):
            for i in range(self.n_qubits):
                qml.Rot(weights[l, i, 0], weights[l, i, 1], weights[l, i, 2], wires=i)

            for i in range(self.n_qubits):
                qml.CNOT(wires=[i, (i + 1) % self.n_qubits])

        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def forward(self, x):
        return self.q_layer(x)

class QuantumProjectionAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_qubits=4, q_layers=1, dropout=0.1):
        """
        Quantum-Classical Attention mechanism.
        Combines classical attention with quantum-enhanced feature extraction.
        """
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.n_qubits = n_qubits

        self.pre_q_proj = nn.Linear(self.head_dim, n_qubits)
        self.pre_k_proj = nn.Linear(self.head_dim, n_qubits)

        try:
            print("using lighting")
            import pennylane_lightning
            self.dev_type = "lightning.qubit"
        except ImportError:
            print("NOT using lighting")
            self.dev_type = "default.qubit"
            
        self.q_circuit = QuantumVariationalLayer(n_qubits, q_layers)
        self.k_circuit = QuantumVariationalLayer(n_qubits, q_layers)

        self.post_q_proj = nn.Linear(n_qubits, self.head_dim)
        self.post_k_proj = nn.Linear(n_qubits, self.head_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        B, L, H, D = queries.shape
        _, S, _, _ = keys.shape 
        q = queries 
        k = keys
        v = values
        q_q = self.pre_q_proj(q)
        k_q = self.pre_k_proj(k)

        q_flat = q_q.reshape(-1, self.n_qubits) 
        k_flat = k_q.reshape(-1, self.n_qubits)
        
        q_flat = torch.tanh(q_flat)
        k_flat = torch.tanh(k_flat)

        q_out_flat = self.q_circuit(q_flat)
        k_out_flat = self.k_circuit(k_flat)

        q_out = self.post_q_proj(q_out_flat).view(B, L, H, D)
        k_out = self.post_k_proj(k_out_flat).view(B, S, H, D)
        
        q_out = q_out.permute(0, 2, 1, 3)
        k_out = k_out.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3) 
        scores = torch.matmul(q_out, k_out.transpose(-2, -1)) * self.scale

        if attn_mask is not None:

            scores = scores.masked_fill(attn_mask == 0, -1e9)

        attn = self.dropout(torch.softmax(scores, dim=-1))
        

        context = torch.matmul(attn, v) 
        
        context = context.permute(0, 2, 1, 3).contiguous()
        
        return context, attn

class QuantumProjectionAttentionHybrid(nn.Module):
    def __init__(self, d_model, n_heads, n_qubits=4, q_layers=1, dropout=0.1):
        """
        Quantum-Classical Attention mechanism with Hybrid Q/K.
        Combines quantum-enhanced key/queries with classical keys/queries via concatenation.
        """
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.n_qubits = n_qubits

        self.pre_q_proj = nn.Linear(self.head_dim, n_qubits)
        self.pre_k_proj = nn.Linear(self.head_dim, n_qubits)

        self.q_circuit = QuantumVariationalLayer(n_qubits, q_layers)
        self.k_circuit = QuantumVariationalLayer(n_qubits, q_layers)

        self.post_q_proj = nn.Linear(self.head_dim + n_qubits, self.head_dim)
        self.post_k_proj = nn.Linear(self.head_dim + n_qubits, self.head_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, queries, keys, values, attn_mask=None):
        B, L, H, D = queries.shape
        _, S, _, _ = keys.shape
        q = queries
        k = keys
        v = values

        q_q = self.pre_q_proj(q) 
        k_q = self.pre_k_proj(k) 

        q_flat = torch.tanh(q_q.reshape(-1, self.n_qubits))
        k_flat = torch.tanh(k_q.reshape(-1, self.n_qubits))
        q_out_flat = self.q_circuit(q_flat)
        k_out_flat = self.k_circuit(k_flat)
        q_out = q_out_flat.view(B, L, H, self.n_qubits)
        k_out = k_out_flat.view(B, S, H, self.n_qubits)


        q_hybrid = torch.cat([q, q_out], dim=-1) 
        k_hybrid = torch.cat([k, k_out], dim=-1)
        q_proj = self.post_q_proj(q_hybrid)
        k_proj = self.post_k_proj(k_hybrid)
        q_proj = q_proj.permute(0,2,1,3)
        k_proj = k_proj.permute(0,2,1,3)
        v_proj = v.permute(0,2,1,3) 

        scores = torch.matmul(q_proj, k_proj.transpose(-2,-1)) * self.scale
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, -1e9)

        attn = self.dropout(torch.softmax(scores, dim=-1))
        context = torch.matmul(attn, v_proj)
        context = context.permute(0,2,1,3).contiguous()

        return context, attn

class QuantumKernelAttentionFidelity(nn.Module):
    def __init__(self, d_model, n_heads, n_qubits=4, dropout=0.1, n_layers=2):
        """
        Implementation inspired by Smaldone et al. (Molecular) approach.
        Replaces Hadamard Test similarity mechanism with quantum fidelity mechanism.
        """
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.n_qubits = n_qubits
        self.n_layers = n_layers # Depth of the quantum circuit
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Projections to latent quantum space
        self.q_proj = nn.Linear(self.head_dim, n_qubits)
        self.k_proj = nn.Linear(self.head_dim, n_qubits)

        self.dropout = nn.Dropout(dropout)
        
        self.dev = qml.device("default.qubit", wires=n_qubits)
        
        self.qnode = qml.QNode(self.fidelity_circuit, self.dev, interface='torch', diff_method="backprop")

    def feature_map(self, angles):
        for l in range(self.n_layers):
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)
                qml.RZ(angles[..., i], wires=i)
            # Entanglement layer
            # creates the non-separable kernel space crucial for molecules
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
                qml.RZ(angles[..., i] * angles[..., i+1], wires=i+1) 
                qml.CNOT(wires=[i, i + 1])

    def feature_map_inverse(self, angles):
        """Inverse of the feature map (Adjoint)."""
        for l in range(self.n_layers - 1, -1, -1):
            for i in range(self.n_qubits - 2, -1, -1):
                qml.CNOT(wires=[i, i + 1])
                qml.RZ(-angles[..., i] * angles[..., i+1], wires=i+1)
                qml.CNOT(wires=[i, i + 1])

            for i in range(self.n_qubits):
                qml.RZ(-angles[..., i], wires=i)
                qml.Hadamard(wires=i) 

    def fidelity_circuit(self, q_angles, k_angles):
        self.feature_map(q_angles)

        self.feature_map_inverse(k_angles)

        # Measure probability of all-zero state
        return qml.probs(wires=range(self.n_qubits))

    def forward(self, queries, keys, values, attn_mask=None):
        B, L, H, D = queries.shape
        _, S, _, _ = keys.shape

        # project inputs to angles
        q_angles = torch.tanh(self.q_proj(queries)) * np.pi
        k_angles = torch.tanh(self.k_proj(keys)) * np.pi

        # Broadcasting preparation
        q_exp = q_angles.permute(0, 2, 1, 3).unsqueeze(3)
        k_exp = k_angles.permute(0, 2, 1, 3).unsqueeze(2)
        
        target_shape = (B, H, L, S, self.n_qubits)
        q_broad = q_exp.expand(target_shape)
        k_broad = k_exp.expand(target_shape)

        total_evals = B * H * L * S
        q_flat = q_broad.reshape(total_evals, self.n_qubits)
        k_flat = k_broad.reshape(total_evals, self.n_qubits)

        # compute quantum attention scores (Fidelity)
        probs = self.qnode(q_flat, k_flat)
        
        # measurement of 0 state
        attn_scores_flat = probs[:, 0]
        
        attn_scores_flat = attn_scores_flat.type_as(queries)
        attn_scores = attn_scores_flat.view(B, H, L, S)

        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask == 0, 0) 

        attn_weights = attn_scores / (attn_scores.sum(dim=-1, keepdim=True) + 1e-6)
        
        attn_weights = self.dropout(attn_weights)

        v = values.permute(0, 2, 1, 3)
        
        context = torch.matmul(attn_weights, v) 
        
        context = context.permute(0, 2, 1, 3).contiguous() 

        return context, attn_weights


class QuantumKernelAttentionHadamard(nn.Module):
    def __init__(self, d_model, n_heads, n_qubits=4, n_layers=1, dropout=0.1):
        """
        Pennylane Implementation inspired by Smaldone et al. (Molecular) approach.
        Uses Hadamard Test similarity mechanism with ancilla qubit.
        EXTREMELY SLOW compared to fidelity approach.
        """
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.n_qubits = n_qubits
        self.n_layers = n_layers 
        
        self.n_params = n_qubits * n_layers
        self.q_proj = nn.Linear(self.head_dim, self.n_params)
        self.k_proj = nn.Linear(self.head_dim, self.n_params)

        self.dropout = nn.Dropout(dropout)
        
        try:
            self.dev = qml.device("lightning.qubit", wires=n_qubits + 1) # for ancilla qubit
        except:
            self.dev = qml.device("default.qubit", wires=n_qubits + 1)
            
        #self.qnode = qml.QNode(self.hadamard_test_circuit, self.dev, interface='torch', diff_method="backprop")
        self.qnode = qml.QNode(self.hadamard_test_circuit, self.dev, interface='torch', diff_method="adjoint")


    def ansatz(self, params, wires):
        shape = params.shape
        new_shape = shape[:-1] + (self.n_layers, self.n_qubits)
        params = params.reshape(new_shape)

        for l in range(self.n_layers):
            for i in range(self.n_qubits):
                qml.RY(params[..., l, i], wires=wires[i])
            
            # Entanglement Layer (Ring CNOTs)
            if self.n_qubits > 1:
                for i in range(self.n_qubits):
                    qml.CNOT(wires=[wires[i], wires[(i + 1) % self.n_qubits]])

    def hadamard_test_circuit(self, q_params, k_params):

        ancilla = self.n_qubits  # idc of ancilla wire
        data_wires = range(self.n_qubits)

        # initialize ancilla
        qml.Hadamard(wires=ancilla)

        qml.ctrl(self.ansatz, control=ancilla)(q_params, wires=data_wires)
        qml.adjoint(qml.ctrl(self.ansatz, control=ancilla))(k_params, wires=data_wires)

        # final interference
        qml.Hadamard(wires=ancilla)
        return qml.expval(qml.PauliZ(wires=ancilla))

    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        B, L, H, D = queries.shape
        _, S, _, _ = keys.shape

        q_params = torch.tanh(self.q_proj(queries)) * np.pi 
        k_params = torch.tanh(self.k_proj(keys)) * np.pi

        q_broad = q_params.permute(0, 2, 1, 3).unsqueeze(3)
        k_broad = k_params.permute(0, 2, 1, 3).unsqueeze(2)
        
        target_shape = (B, H, L, S, self.n_params)
        q_broad = q_broad.expand(target_shape)
        k_broad = k_broad.expand(target_shape)
        
        q_flat = q_broad.reshape(-1, self.n_params)
        k_flat = k_broad.reshape(-1, self.n_params)

        attn_scores_flat = self.qnode(q_flat, k_flat)
        
        attn_scores_flat = attn_scores_flat.type_as(queries)
        
        attn_scores = attn_scores_flat.view(B, H, L, S)

        attn_scores = attn_scores / math.sqrt(self.head_dim)

        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask == 0, -1e9)

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        v = values.permute(0, 2, 1, 3)
        context = torch.matmul(attn_weights, v).permute(0, 2, 1, 3).contiguous()

        return context, attn_weights

