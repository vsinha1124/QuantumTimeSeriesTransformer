import pennylane as qml
import torch
import torch.nn as nn
import numpy as np

def ansatz_14_pennylane(n_qubits: int, layers: int = 1):
    """
    PennyLane implementation of 'ansatz 14' from Sim et al. (2019).
    Total parameters = 4 * n_qubits * layers.
    """
    def circuit(params, wires):
        param_idx = 0
        for _ in range(layers):
            # First RY layer
            for i in range(n_qubits):
                qml.RY(params[param_idx], wires=wires[i])
                param_idx += 1
            # First CRX ring (reverse order)
            for i in range(n_qubits - 1, -1, -1):
                qml.CRX(params[param_idx], wires=[wires[i], wires[(i + 1) % n_qubits]])
                param_idx += 1
            # Second RY layer
            for i in range(n_qubits):
                qml.RY(params[param_idx], wires=wires[i])
                param_idx += 1
            # Second CRX ring (forward-ish)
            order = [n_qubits - 1] + list(range(n_qubits - 1))
            for i in order:
                qml.CRX(params[param_idx], wires=[wires[i], wires[(i - 1) % n_qubits]])
                param_idx += 1
        return param_idx  # number of params used
    return circuit

class QuixerCore(nn.Module):
    """
    Core Quixer block for a single context (one sequence per batch),
    implemented with PennyLane + classical statevector math.

    Input: tokens_emb [L, d_model]  (sequence of patch/token embeddings)
    Output: measurement_features [3 * n_qubits]
    """

    def __init__(self, n_qubits: int, n_tokens: int, d_model: int,
                 qsvt_degree: int = 2, n_ansatz_layers: int = 1,
                 dev_name: str = "default.qubit"):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_tokens = n_tokens
        self.d_model = d_model
        self.degree = qsvt_degree
        self.n_ansatz_layers = n_ansatz_layers

        # Number of parameters in ansatz-14
        self.n_pqc_params = 4 * n_qubits * n_ansatz_layers

        # Map token embeddings -> PQC parameters
        self.embedding_to_angles = nn.Linear(d_model, self.n_pqc_params)

        # QSVT polynomial coefficients [c_0, ..., c_degree]
        self.qsvt_coeffs = nn.Parameter(
            torch.randn(self.degree + 1) * 0.1
        )

        # LCU coefficients per token (complex)
        self.lcu_coeffs = nn.Parameter(
            torch.randn(self.n_tokens, dtype=torch.cfloat)
        )

        # Feedforward PQC parameters (same ansatz shape)
        self.ff_params = nn.Parameter(
            torch.randn(self.n_pqc_params)
        )

        # PennyLane device
        try:
            self.dev = qml.device(dev_name, wires=self.n_qubits)
        except Exception:
            self.dev = qml.device("default.qubit", wires=self.n_qubits)

        self.ansatz = ansatz_14_pennylane(self.n_qubits, self.n_ansatz_layers)

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def pqc_state(params, init_state):
            """Apply ansatz-14 to an initial state and return final statevector."""
            qml.QubitStateVector(init_state, wires=range(self.n_qubits))
            self.ansatz(params, wires=range(self.n_qubits))
            return qml.state()

        self.q_pqc_state = pqc_state

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def pqc_measure(params, init_state):
            """Apply feedforward PQC then measure X, Y, Z on all qubits."""
            qml.QubitStateVector(init_state, wires=range(self.n_qubits))
            self.ansatz(params, wires=range(self.n_qubits))
            obs = []
            for w in range(self.n_qubits):
                obs.append(qml.expval(qml.PauliX(w)))
            for w in range(self.n_qubits):
                obs.append(qml.expval(qml.PauliY(w)))
            for w in range(self.n_qubits):
                obs.append(qml.expval(qml.PauliZ(w)))
            return obs  # list of 3*n_qubits expectation values

        self.q_pqc_measure = pqc_measure

    def _apply_lcu(self, monomial_state, pqc_params_per_token, lcu_coeffs):
        """
        Apply classical LCU: sum_t b_t U_t |phi>.
        monomial_state: [2^n_qubits] complex vector
        pqc_params_per_token: [L_used, n_pqc_params]
        lcu_coeffs: [L_used] complex
        """
        L_used = pqc_params_per_token.shape[0]
        # Normalize state for the circuit; track amplitude separately
        norm = torch.norm(monomial_state)
        if norm < 1e-8:
            # fallback to |0...0>
            norm = monomial_state.new_tensor(1.0)
            init_state = torch.zeros_like(monomial_state)
            init_state[0] = 1.0
        else:
            init_state = monomial_state / norm

        lcu_state = torch.zeros_like(monomial_state)
        for t in range(L_used):
            params_t = pqc_params_per_token[t]
            evolved = self.q_pqc_state(params_t, init_state)
            lcu_state = lcu_state + lcu_coeffs[t] * evolved

        # Restore the overall norm scaling
        return norm * lcu_state

    def forward(self, tokens_emb: torch.Tensor) -> torch.Tensor:
        """
        tokens_emb: [L, d_model], L <= n_tokens (context length).
        Returns: [3 * n_qubits] expectation values.
        """
        device = tokens_emb.device
        L = tokens_emb.shape[0]
        L_used = min(L, self.n_tokens)

        # Project embeddings to PQC angles for the first L_used tokens
        pqc_angles = self.embedding_to_angles(tokens_emb[:L_used])  # [L_used, n_pqc_params]

        # Normalize LCU coefficients (per sequence)
        lcu_coeffs = self.lcu_coeffs[:L_used]
        # L1 normalization (like Quixer)
        lcu_coeffs = lcu_coeffs / torch.clamp(
            torch.sum(torch.abs(lcu_coeffs)), min=1e-8
        )

        # Initial |0...0> state
        dim = 2 ** self.n_qubits
        init_state = torch.zeros(dim, dtype=torch.cfloat, device=device)
        init_state[0] = 1.0

        # QSVT polynomial p(M) = c_0 I + c_1 M + c_2 M^2 + ...
        coeffs = self.qsvt_coeffs
        acc_state = coeffs[0] * init_state
        monomial_state = init_state

        for k in range(1, len(coeffs)):
            monomial_state = self._apply_lcu(monomial_state, pqc_angles, lcu_coeffs)
            acc_state = acc_state + coeffs[k] * monomial_state

        # Normalize by L1 norm of polynomial coefficients (as in Quixer)
        poly_norm = torch.norm(coeffs, p=1)
        acc_state = acc_state / torch.clamp(poly_norm, min=1e-8)

        # Feedforward PQC + measurement
        # Normalize state for the device again
        state_norm = torch.norm(acc_state)
        if state_norm < 1e-8:
            state_norm = acc_state.new_tensor(1.0)
            ff_in = torch.zeros_like(acc_state)
            ff_in[0] = 1.0
        else:
            ff_in = acc_state / state_norm

        exps = self.q_pqc_measure(self.ff_params, ff_in)  # list length = 3 * n_qubits
        exps = torch.stack(exps)  # [3 * n_qubits]
        return exps

class QuixerAttentionLayer_OptionA(nn.Module):
    """
    Option A: Replace attention completely by a Quixer global mixer.
    Compatible with your EncoderLayer interface: takes (q, k, v) and returns (out, attn).

    out[b, l, :] = v[b, l, :] + W * QuixerCore(v[b, :, :])
    """

    def __init__(
        self,
        d_model: int,
        n_qubits: int = 4,
        n_tokens: int = 32,         # max context window for Quixer
        qsvt_degree: int = 2,
        n_ansatz_layers: int = 1,
        dev_name: str = "default.qubit",
        output_attention: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_qubits = n_qubits
        self.n_tokens = n_tokens
        self.output_attention = output_attention

        self.quixer_core = QuixerCore(
            n_qubits=n_qubits,
            n_tokens=n_tokens,
            d_model=d_model,
            qsvt_degree=qsvt_degree,
            n_ansatz_layers=n_ansatz_layers,
            dev_name=dev_name,
        )

        # Map measurement features back to d_model
        self.measure_to_dmodel = nn.Linear(3 * n_qubits, d_model)

    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        """
        queries: [B, L, D]
        keys:   [B, L, D]  (ignored)
        values: [B, L, D]
        """
        B, L, D = values.shape
        device = values.device

        # For each batch element, run QuixerCore on its sequence of values
        global_vecs = []
        for b in range(B):
            tokens_emb = values[b]  # [L, D]
            exps = self.quixer_core(tokens_emb)  # [3 * n_qubits]
            # Convert to float32 to match model dtype
            global_vec = self.measure_to_dmodel(exps.float())  # [D]
            global_vecs.append(global_vec)

        global_vecs = torch.stack(global_vecs, dim=0)  # [B, D]
        # Broadcast to all positions
        global_vecs_expanded = global_vecs.unsqueeze(1).expand(B, L, D)  # [B, L, D]

        # Option A: output = values + global_context
        out = values + global_vecs_expanded

        attn = None  # we have no meaningful attention map here
        if self.output_attention:
            # You can create a dummy attention tensor if your training loop expects it
            attn = torch.zeros(B, 1, L, L, device=device)

        return out, attn
