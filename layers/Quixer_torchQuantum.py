import torch
import torch.nn as nn
import torchquantum as tq
from torchquantum import GeneralEncoder, QuantumDevice
from math import log2
import itertools
def ansatz_14_torchquantum_spec(n_qubits: int, layers: int = 1):
    enc = []
    counter = itertools.count(0)

    for _ in range(layers):
        # RY layer
        enc.extend([
            {"input_idx": [next(counter)], "func": "ry", "wires": [i]}
            for i in range(n_qubits)
        ])
        # CRX reverse ring
        enc.extend([
            {"input_idx": [next(counter)], "func": "crx", "wires": [i, (i+1) % n_qubits]}
            for i in range(n_qubits - 1, -1, -1)
        ])
        # RY again
        enc.extend([
            {"input_idx": [next(counter)], "func": "ry", "wires": [i]}
            for i in range(n_qubits)
        ])
        # CRX forward ring
        enc.extend([
            {"input_idx": [next(counter)], "func": "crx", "wires": [i, (i-1) % n_qubits]}
            for i in [n_qubits - 1] + list(range(n_qubits - 1))
        ])

    return enc

class FusedAnsatz14(torch.nn.Module):
    def __init__(self, n_qubits, layers):
        super().__init__()
        self.n_qubits = n_qubits
        self.layers = layers

    def forward(self, qdev, params):
        B = params.shape[0]
        for b in range(B):
            idx = 0
            for _ in range(self.layers):
                # RY layer
                for w in range(self.n_qubits):
                    op = tq.RY(has_params=True, trainable=True, wires=w, init_params=params[b, idx].item())
                    qdev.apply_op(op, bsz_id=b)
                    idx += 1

                # CRX reverse
                for w in reversed(range(self.n_qubits)):
                    op = tq.CRX(has_params=True, trainable=True, wires=[w, (w + 1) % self.n_qubits], init_params=params[b, idx].item())
                    qdev.apply_op(op, bsz_id=b)
                    idx += 1

                # RY
                for w in range(self.n_qubits):
                    op = tq.RY(has_params=True, trainable=True, wires=w, init_params=params[b, idx].item())
                    qdev.apply_op(op, bsz_id=b)
                    idx += 1

                # CRX forward
                order = [self.n_qubits - 1] + list(range(self.n_qubits - 1))
                for w in order:
                    op = tq.CRX(has_params=True, trainable=True, wires=[w, (w - 1) % self.n_qubits], init_params=params[b, idx].item())
                    qdev.apply_op(op, bsz_id=b)
                    idx += 1

def apply_lcu_fast(state, pqc_params, pqc, qdev, lcu):
    # state: [B, 2^n]
    # pqc_params: [B, L, P]
    B, L, P = pqc_params.shape
    n = state.shape[-1]

    # Expand state -> [B*L, 2^n]
    expanded = state.unsqueeze(1).repeat(1, L, 1).reshape(B*L, n)
    qdev.set_states(expanded)

    # Flatten parameters for all unitaries
    flat_params = pqc_params.reshape(B * L, P)

    # Apply PQC to all tokens in parallel
    pqc(qdev, flat_params)

    # Extract evolved states and reshape back to [B, L, 2^n]
    evolved = qdev.get_states_1d().reshape(B, L, n)

    # Compute LCU: multiply across L in one CUDA kernel
    return torch.einsum("bln,bl->bn", evolved, lcu)


def apply_lcu(
    initial_states,
    pqc_parameters,
    pqc,
    tq_device,
    n_qubits,
    lcu_coeffs
):
    batch = initial_states.shape[0]
    L = pqc_parameters.shape[1]

    # Repeat initial states for all tokens
    repeated_state = initial_states.repeat(1, L).reshape(batch * L, 2**n_qubits)

    # Load into TorchQuantum
    tq_device.set_states(repeated_state)

    # Apply PQC to all token states at once
    pqc(tq_device, pqc_parameters.reshape(batch * L, -1))

    # Extract statevector
    states = tq_device.get_states_1d().reshape(batch, L, 2**n_qubits)

    # Sum over token dimension using LCU coefficients
    return torch.einsum("bld,bl->bd", states, lcu_coeffs)

def apply_qsvt_fast(state, pqc_params, pqc, qdev, lcu, coeffs):
    # state: [B, 2^n]
    B = state.shape[0]
    acc = coeffs[0] * state
    mono = state

    for k in range(1, coeffs.shape[0]):
        mono = apply_lcu_fast(mono, pqc_params, pqc, qdev, lcu)
        acc = acc + coeffs[k] * mono

    return acc / torch.norm(coeffs, p=1)


def apply_qsvt(
    initial_states,
    pqc_params,
    pqc,
    tq_device,
    n_qubits,
    lcu_coeffs,
    qsvt_coeffs
):
    batch = initial_states.shape[0]

    # QSVT polynomial accumulation
    acc = qsvt_coeffs[0] * initial_states
    mono = initial_states

    for c in qsvt_coeffs[1:]:
        mono = apply_lcu(mono, pqc_params, pqc, tq_device, n_qubits, lcu_coeffs)
        acc = acc + c * mono

    # Normalization by L1 norm of coefficients
    return acc / torch.norm(qsvt_coeffs, p=1)
class QuixerCore_TQ(nn.Module):
    def __init__(self, n_qubits, n_tokens, d_model, qsvt_degree=2, n_ansatz_layers=1, device="cuda"):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_tokens = n_tokens
        self.d_model = d_model
        self.degree = qsvt_degree
        self.device = device

        self.n_pqc = 4 * n_qubits * n_ansatz_layers

        # Embedding → angles
        self.embedding_to_angles = nn.Linear(d_model, self.n_pqc)

        # LCU coefficients
        self.lcu_coeffs = nn.Parameter(torch.randn(n_tokens, dtype=torch.complex64))

        # QSVT polynomial coefficients c0..cD
        self.qsvt_coeffs = nn.Parameter(torch.randn(self.degree + 1))

        # TorchQuantum device
        self.tq_device = QuantumDevice(n_wires=n_qubits, bsz=1, device=device)

        # PQC for token unitaries
        self.token_pqc = GeneralEncoder(ansatz_14_torchquantum_spec(n_qubits, n_ansatz_layers))
        self.token_pqc.n_wires = n_qubits

        # PQC for feedforward step
        self.ff_params = nn.Parameter(torch.randn(self.n_pqc))
        self.ff_pqc = GeneralEncoder(ansatz_14_torchquantum_spec(n_qubits, 1))
        self.ff_pqc.n_wires = n_qubits

        self.measure = tq.MeasureMultipleTimes(
            [
                {"wires": range(n_qubits), "observables": ["x"] * n_qubits},
                {"wires": range(n_qubits), "observables": ["y"] * n_qubits},
                {"wires": range(n_qubits), "observables": ["z"] * n_qubits},
            ]
        )

    def forward(self, tokens_emb):
        # tokens_emb: [B, L, D]
        B, L, D = tokens_emb.shape
        L = min(L, self.n_tokens)

        # Recreate device for batch size
        self.tq_device = QuantumDevice(n_wires=self.n_qubits, bsz=B, device=self.device)

        # Map embeddings to angles
        pqc_params = self.embedding_to_angles(tokens_emb[:, :L])    # [B, L, n_pqc]

        # Normalize LCU
        lcu = self.lcu_coeffs[:L]
        lcu = lcu / torch.sum(torch.abs(lcu))

        init_state = torch.zeros(B, 2**self.n_qubits, dtype=torch.complex64, device=self.device)
        init_state[:, 0] = 1.0

        # QSVT+LCU state
        qsvt_state = apply_qsvt(
            init_state,
            pqc_params,
            self.token_pqc,
            self.tq_device,
            self.n_qubits,
            lcu.unsqueeze(0).expand(B, -1),
            self.qsvt_coeffs
        )

        qsvt_state = qsvt_state / torch.norm(qsvt_state, dim=-1, keepdim=True)

        # Load, apply FF PQC
        self.tq_device.set_states(qsvt_state)
        self.ff_pqc(self.tq_device, self.ff_params.unsqueeze(0).expand(B, -1))

        # Measure X,Y,Z
        out = self.measure(self.tq_device)        # [B, 3, n_qubits]
        out = out.reshape(B, -1)  # [B, 3*n_qubits]
        return out.float()
        
class QuixerAttentionLayer_OptionC(nn.Module):
    def __init__(self, d_model, n_qubits=4, n_tokens=96, qsvt_degree=2, n_ansatz_layers=1, device="cuda"):
        super().__init__()
        self.d_model = d_model

        self.core = QuixerCore_TQ(
            n_qubits=n_qubits,
            n_tokens=n_tokens,
            d_model=d_model,
            qsvt_degree=qsvt_degree,
            n_ansatz_layers=n_ansatz_layers,
            device=device
        )

        self.linear = nn.Linear(3 * n_qubits, d_model)

    # def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
    #     B, L, D = values.shape
    #     outputs = []

    #     for b in range(B):
    #         qvec = self.core(values[b])     # [3*n_qubits]
    #         outputs.append(self.linear(qvec))  # [D]

    #     global_vecs = torch.stack(outputs).unsqueeze(1).expand(B, L, D)
    #     out = values + global_vecs
    #     return out, None
    def forward(self, Q, K, V, attn_mask=None, tau=None, delta=None):
        B, L, D = V.shape

        # Vectorized: evolve all B samples at once
        exps = self.core(V)  # [B, 3*n_qubits]
        exps = self.linear(exps)  # [B, D]

        return V + exps.unsqueeze(1).expand(-1, L, -1), None
