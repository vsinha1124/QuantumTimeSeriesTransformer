import pennylane as qml
import torch
import torch.nn as nn
import numpy as np


def ansatz_14_pennylane(n_qubits: int, layers: int = 1):
    """
    PennyLane implementation of 'ansatz 14' from Sim et al. (2019).
    Total parameters = 4 * n_qubits * layers.
    
    IMPORTANT: This function expects params to be a 1D array when called.
    When used with @qml.batch_params, PennyLane will handle the batching
    and call this function with each sample's parameters separately.
    """
    def circuit(params, wires):
        # Convert to numpy array to ensure proper indexing
        if isinstance(params, torch.Tensor):
            params_array = params.detach().cpu().numpy()
        else:
            params_array = np.array(params)
        
        # Flatten in case it has extra dimensions
        params_flat = params_array.flatten()
        
        param_idx = 0
        for _ in range(layers):
            # First RY layer
            for i in range(n_qubits):
                qml.RY(float(params_flat[param_idx]), wires=wires[i])
                param_idx += 1
            # First CRX ring (reverse order)
            for i in range(n_qubits - 1, -1, -1):
                qml.CRX(float(params_flat[param_idx]), wires=[wires[i], wires[(i + 1) % n_qubits]])
                param_idx += 1
            # Second RY layer
            for i in range(n_qubits):
                qml.RY(float(params_flat[param_idx]), wires=wires[i])
                param_idx += 1
            # Second CRX ring (forward style)
            order = [n_qubits - 1] + list(range(n_qubits - 1))
            for i in order:
                qml.CRX(float(params_flat[param_idx]), wires=[wires[i], wires[(i - 1) % n_qubits]])
                param_idx += 1
        
        # Verify we used the right number of parameters
        expected_params = 4 * n_qubits * layers
        if param_idx != expected_params:
            raise ValueError(
                f"Parameter count mismatch: used {param_idx}, expected {expected_params}"
            )

        return param_idx

    return circuit


class QuixerCore(nn.Module):
    """
    QuixerCore using PennyLane statevector simulation with proper batching.
    """

    def __init__(
        self,
        n_qubits: int,
        n_tokens: int,
        d_model: int,
        qsvt_degree: int = 2,
        n_ansatz_layers: int = 1,
        dev_name: str = "default.qubit",
    ):
        super().__init__()

        self.n_qubits = n_qubits
        self.n_tokens = n_tokens
        self.d_model = d_model
        self.degree = qsvt_degree
        self.n_ansatz_layers = n_ansatz_layers
        self.dim = 2 ** n_qubits
        self.n_pqc_params = self.n_ansatz_layers * 4 * self.n_qubits

        # Map token embeddings -> ansatz angles
        self.embedding_to_angles = nn.Linear(d_model, self.n_pqc_params)

        # QSVT polynomial coefficients
        self.qsvt_coeffs = nn.Parameter(torch.randn(self.degree + 1) * 0.1)

        # LCU coefficients per token (complex)
        self.lcu_coeffs = nn.Parameter(torch.randn(self.n_tokens, dtype=torch.cdouble))

        # Feedforward PQC parameters
        self.ff_params = nn.Parameter(torch.randn(self.n_pqc_params))

        # PennyLane device selection
        try:
            self.dev = qml.device(dev_name, wires=self.n_qubits, shots=None)
        except Exception:
            try:
                self.dev = qml.device("lightning.gpu", wires=self.n_qubits, shots=None)
            except Exception:
                try:
                    self.dev = qml.device("lightning.qubit", wires=self.n_qubits, shots=None)
                except Exception:
                    self.dev = qml.device("default.qubit", wires=self.n_qubits, shots=None)

        self.ansatz = ansatz_14_pennylane(self.n_qubits, self.n_ansatz_layers)

        # Define individual evolution QNode (we'll call it in a loop)
        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def evolve_single(single_params, single_init_state):
            """
            Process a single set of parameters and initial state.
            
            Args:
                single_params: [n_pqc_params]
                single_init_state: [2**n_qubits]
            
            Returns:
                [2**n_qubits] evolved state
            """
            qml.QubitStateVector(single_init_state, wires=range(self.n_qubits))
            self.ansatz(single_params, wires=range(self.n_qubits))
            return qml.state()

        self.q_evolve_single = evolve_single

        # Define measurement QNode
        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def measure_pqc(params, init_state):
            qml.QubitStateVector(init_state, wires=range(self.n_qubits))
            self.ansatz(params, wires=range(self.n_qubits))
            
            obs = []
            # X measurements
            for w in range(self.n_qubits):
                obs.append(qml.expval(qml.PauliX(w)))
            # Y measurements
            for w in range(self.n_qubits):
                obs.append(qml.expval(qml.PauliY(w)))
            # Z measurements
            for w in range(self.n_qubits):
                obs.append(qml.expval(qml.PauliZ(w)))
            
            return obs

        self.q_measure_pqc = measure_pqc

    def _apply_lcu_pennylane(
        self,
        monomial_state: torch.Tensor,
        pqc_params_per_token: torch.Tensor,
        lcu_coeffs: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply LCU using batched quantum evolution.
        
        Args:
            monomial_state: [2**n_qubits]
            pqc_params_per_token: [L_used, n_pqc_params]
            lcu_coeffs: [L_used] complex coefficients
        
        Returns:
            [2**n_qubits] resulting state
        """
        device = monomial_state.device
        L_used = pqc_params_per_token.shape[0]

        # Normalize initial state
        norm = torch.norm(monomial_state)
        if norm < 1e-8:
            norm = torch.tensor(1.0, device=device)
            init_state = torch.zeros_like(monomial_state)
            init_state[0] = 1.0
        else:
            init_state = monomial_state / norm

        # Ensure params have correct shape [L_used, n_pqc_params]
        assert pqc_params_per_token.shape == (L_used, self.n_pqc_params), \
            f"Expected shape [{L_used}, {self.n_pqc_params}], got {pqc_params_per_token.shape}"

        # Process each token's evolution sequentially
        evolved_states = []
        for i in range(L_used):
            single_params = pqc_params_per_token[i]  # [n_pqc_params]
            evolved_state = self.q_evolve_single(single_params, init_state)
            evolved_states.append(evolved_state)
        
        # Stack results: [L_used, dim]
        evolved_states = torch.stack(evolved_states)

        # Linear combination with LCU coefficients
        lcu_state = torch.einsum("td,t->d", evolved_states, lcu_coeffs)

        return norm * lcu_state

    def forward(self, tokens_emb: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through QuixerCore.
        
        Args:
            tokens_emb: [L, d_model] token embeddings
        
        Returns:
            [3 * n_qubits] measurement expectation values
        """
        device = tokens_emb.device
        L = tokens_emb.shape[0]
        L_used = min(L, self.n_tokens)

        # Project embeddings to PQC angles
        pqc_angles = self.embedding_to_angles(tokens_emb[:L_used])  # [L_used, n_pqc_params]

        # Normalize LCU coefficients
        lcu_coeffs = self.lcu_coeffs[:L_used]
        lcu_coeffs = lcu_coeffs / torch.clamp(torch.sum(torch.abs(lcu_coeffs)), min=1e-8)

        # Initial |0...0> state
        dim = 2 ** self.n_qubits
        init_state = torch.zeros(dim, dtype=torch.complex128, device=device)
        init_state[0] = 1.0

        # QSVT polynomial evaluation
        coeffs = self.qsvt_coeffs
        acc_state = coeffs[0] * init_state
        monomial_state = init_state

        for k in range(1, len(coeffs)):
            monomial_state = self._apply_lcu_pennylane(
                monomial_state,
                pqc_angles,
                lcu_coeffs,
            )
            acc_state = acc_state + coeffs[k] * monomial_state

        # Normalize by polynomial L1 norm
        poly_norm = torch.norm(coeffs, p=1)
        acc_state = acc_state / torch.clamp(poly_norm, min=1e-8)

        # Normalize before feedforward PQC
        state_norm = torch.norm(acc_state)
        if state_norm < 1e-8:
            state_norm = torch.tensor(1.0, device=device)
            ff_in = torch.zeros_like(acc_state)
            ff_in[0] = 1.0
        else:
            ff_in = acc_state / state_norm

        # Apply feedforward PQC and measure
        exps = self.q_measure_pqc(self.ff_params, ff_in)
        exps = torch.stack(exps)  # [3 * n_qubits]

        return exps


class QuixerAttentionLayer_OptionB(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_qubits: int = 4,
        n_tokens: int = 96,
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

        self.measure_to_dmodel = nn.Linear(3 * n_qubits, d_model)

    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        B, L, D = values.shape
        device = values.device

        global_vecs = []
        for b in range(B):
            tokens_emb = values[b]  # [L, D]
            exps = self.quixer_core(tokens_emb)  # [3 * n_qubits]
            global_vec = self.measure_to_dmodel(exps.float())  # [D]
            global_vecs.append(global_vec)

        global_vecs = torch.stack(global_vecs, dim=0)  # [B, D]
        global_vecs_expanded = global_vecs.unsqueeze(1).expand(B, L, D)

        out = values + global_vecs_expanded
        
        attn = None
        if self.output_attention:
            attn = torch.zeros(B, 1, L, L, device=device)

        return out, attn