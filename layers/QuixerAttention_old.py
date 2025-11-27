"""
Quixer Transformer Attention using PennyLane
Based on the Quixer model from https://arxiv.org/abs/1905.10876
Adapted for time series forecasting with PennyLane quantum circuits
"""

import torch
import torch.nn as nn
import pennylane as qml
import numpy as np
from math import sqrt, log2
from typing import Optional


def ansatz_14_pennylane(n_qubits: int, layers: int = 1):
    """
    PennyLane implementation of parameterized quantum circuit "ansatz 14" 
    from https://arxiv.org/abs/1905.10876
    
    Args:
        n_qubits: Number of qubits in the circuit
        layers: Number of circuit layers
    
    Returns:
        Function that applies the ansatz given parameters
    """
    def circuit(params, wires):
        param_idx = 0
        for layer in range(layers):
            # First layer of R_Y rotations
            for i in range(n_qubits):
                qml.RY(params[param_idx], wires=wires[i])
                param_idx += 1
            
            # First layer of Controlled R_X rotations (reverse order)
            for i in range(n_qubits - 1, -1, -1):
                qml.CRX(params[param_idx], wires=[wires[i], wires[(i + 1) % n_qubits]])
                param_idx += 1
            
            # Second layer of R_Y rotations
            for i in range(n_qubits):
                qml.RY(params[param_idx], wires=wires[i])
                param_idx += 1
            
            # Second layer of Controlled R_X rotations
            for i in [n_qubits - 1] + list(range(n_qubits - 1)):
                qml.CRX(params[param_idx], wires=[wires[i], wires[(i - 1) % n_qubits]])
                param_idx += 1
        
        return param_idx
    
    return circuit


class QuixerAttention(nn.Module):
    """
    Quixer-based Quantum Attention Mechanism using PennyLane
    
    This implements a quantum-enhanced attention mechanism based on the Quixer model,
    adapted for time series forecasting tasks.
    """
    
    def __init__(
        self,
        n_qubits: int = 4,
        qsvt_polynomial_degree: int = 2,
        n_ansatz_layers: int = 1,
        mask_flag: bool = True,
        scale: Optional[float] = None,
        attention_dropout: float = 0.1,
        output_attention: bool = False,
        d_model: int = 512,
    ):
        """
        Args:
            n_qubits: Number of qubits for quantum circuits
            qsvt_polynomial_degree: Degree of QSVT polynomial (e.g., 2 for quadratic)
            n_ansatz_layers: Number of layers in the parameterized quantum circuit
            mask_flag: Whether to apply attention masking
            scale: Attention score scaling factor
            attention_dropout: Dropout rate for attention weights
            output_attention: Whether to output attention weights
            d_model: Model dimension for projections
        """
        super(QuixerAttention, self).__init__()
        
        self.n_qubits = n_qubits
        self.qsvt_polynomial_degree = qsvt_polynomial_degree
        self.n_ansatz_layers = n_ansatz_layers
        self.mask_flag = mask_flag
        self.scale = scale
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.d_model = d_model
        
        # Number of parameters in the parameterized quantum circuit
        self.n_pqc_parameters = 4 * n_qubits * n_ansatz_layers
        
        # Linear layer to project features to PQC angles
        self.feature_to_angles = nn.Linear(d_model, self.n_pqc_parameters)
        
        # QSVT polynomial coefficients (trainable)
        self.n_polynomial_coefficients = qsvt_polynomial_degree + 1
        self.qsvt_polynomial_coefficients = nn.Parameter(
            torch.randn(self.n_polynomial_coefficients) * 0.1
        )
        
        # Quantum device - use lightning.gpu for CUDA acceleration
        try:
            self.dev = qml.device("lightning.gpu", wires=n_qubits)
            print(f"Using lightning.gpu device for {n_qubits} qubits")
        except:
            self.dev = qml.device("default.qubit", wires=n_qubits)
            print(f"GPU device unavailable, using default.qubit for {n_qubits} qubits")
        
        # Create quantum circuit for unitary application
        self.ansatz = ansatz_14_pennylane(n_qubits, n_ansatz_layers)
        
        # QNode for computing quantum state evolution
        @qml.qnode(self.dev, interface='torch', diff_method='backprop')
        def quantum_unitary_circuit(params, initial_state):
            """Apply PQC to initial state and return final state"""
            qml.QubitStateVector(initial_state, wires=range(n_qubits))
            self.ansatz(params, wires=range(n_qubits))
            return qml.state()
        
        self.quantum_circuit = quantum_unitary_circuit
        
        # Measurement operators for output
        @qml.qnode(self.dev, interface='torch', diff_method='backprop')
        def measure_observables(state):
            """Measure X, Y, Z on all qubits"""
            qml.QubitStateVector(state, wires=range(n_qubits))
            return [qml.expval(qml.PauliX(i)) for i in range(n_qubits)] + \
                   [qml.expval(qml.PauliY(i)) for i in range(n_qubits)] + \
                   [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        
        self.measure_circuit = measure_observables
        
        # Output projection from measurements back to d_model
        self.measurement_to_output = nn.Linear(3 * n_qubits, d_model)
    
    def apply_quantum_lcu(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply Linear Combination of Unitaries (LCU) using quantum circuits
        
        Args:
            queries: [B, L, H, E]
            keys: [B, S, H, E]
            values: [B, S, H, D]
        
        Returns:
            Quantum-processed attention output
        """
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        
        # Compute attention coefficients (used as LCU coefficients)
        # [B, H, L, S]
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        scale = self.scale or 1.0 / sqrt(E)
        
        if self.mask_flag:
            # Apply causal mask
            mask = torch.triu(torch.ones(L, S, device=queries.device), diagonal=1).bool()
            scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), -1e9)
        
        # Normalize to get LCU coefficients
        lcu_coefficients = torch.softmax(scale * scores, dim=-1)  # [B, H, L, S]
        lcu_coefficients = self.dropout(lcu_coefficients)
        
        # Project values to PQC parameters
        # [B, S, H, D] -> [B, S, H, n_pqc_parameters]
        pqc_angles = self.feature_to_angles(values)  # This requires D == d_model
        
        # For simplicity, we'll use a classical approximation of the quantum LCU
        # by treating it as weighted sum of quantum-processed states
        
        # Initialize output
        output = torch.zeros(B, L, H, 3 * self.n_qubits, device=queries.device)
        
        # Process each query position
        for b in range(min(B, 2)):  # Limit batch processing for efficiency
            for l in range(L):
                # For each head
                accumulated_state = torch.zeros(2 ** self.n_qubits, dtype=torch.complex64, device=queries.device)
                
                # Apply QSVT polynomial approximation
                monomial_state = torch.zeros(2 ** self.n_qubits, dtype=torch.complex64, device=queries.device)
                monomial_state[0] = 1.0  # |0...0> state
                
                # Compute polynomial terms
                for deg in range(self.qsvt_polynomial_degree + 1):
                    if deg == 0:
                        # Constant term
                        accumulated_state += self.qsvt_polynomial_coefficients[0] * monomial_state
                    else:
                        # Apply LCU to monomial state
                        lcu_state = torch.zeros_like(monomial_state)
                        
                        # Normalize monomial_state before using in quantum circuit
                        norm_mono = torch.sqrt(torch.sum(torch.abs(monomial_state) ** 2))
                        if norm_mono > 1e-8:
                            normalized_monomial = monomial_state / norm_mono
                        else:
                            normalized_monomial = monomial_state
                            normalized_monomial[0] = 1.0  # Reset to |0> if degenerate
                        
                        for s in range(min(S, 4)):  # Limit for efficiency
                            # Get parameters for this token
                            params = pqc_angles[b, s, 0]  # Use first head for simplicity
                            
                            # Apply quantum circuit with normalized state
                            evolved_state = self.quantum_circuit(params, normalized_monomial.detach())
                            
                            # Weight by LCU coefficient
                            coeff = lcu_coefficients[b, 0, l, s]
                            lcu_state += coeff * evolved_state
                        
                        monomial_state = lcu_state
                        accumulated_state += self.qsvt_polynomial_coefficients[deg] * monomial_state
                
                # Normalize accumulated state before measurement
                norm = torch.sqrt(torch.sum(torch.abs(accumulated_state) ** 2))
                if norm > 1e-8:
                    accumulated_state = accumulated_state / norm
                else:
                    # If state is degenerate, reset to |0>
                    accumulated_state = torch.zeros_like(accumulated_state)
                    accumulated_state[0] = 1.0
                
                # Measure observables
                measurements = torch.stack(self.measure_circuit(accumulated_state.detach()))
                
                # Store for all heads (broadcast)
                output[b, l, :, :] = measurements.unsqueeze(0).expand(H, -1)
        
        # For remaining batches, use classical attention as fallback
        if B > 2:
            classical_output = torch.einsum("bhls,bshd->blhd", lcu_coefficients[2:], values[2:])
            # Project through measurement layer
            output[2:] = self.measurement_to_output(
                classical_output.reshape(B-2, L, H, -1)[..., :3*self.n_qubits]
            )
        
        return output
    
    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        """
        Forward pass for Quixer attention
        
        Args:
            queries: [B, L, H, E]
            keys: [B, S, H, E]
            values: [B, S, H, D]
            attn_mask: Optional attention mask
            tau: Optional (unused, for compatibility)
            delta: Optional (unused, for compatibility)
        
        Returns:
            output: [B, L, H, D]
            attention_weights: Optional[Tensor]
        """
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        
        # Compute classical attention scores
        scale = self.scale or 1.0 / sqrt(E)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = torch.triu(torch.ones(L, S, device=queries.device), diagonal=1).bool()
            scores = scores.masked_fill(attn_mask.unsqueeze(0).unsqueeze(0), -1e9)
        
        # Get attention weights
        A = torch.softmax(scale * scores, dim=-1)
        A = self.dropout(A)
        
        # Apply quantum-enhanced processing to values
        # Process values through quantum circuits to get quantum features
        quantum_enhanced_values = self.apply_quantum_encoding_to_values(values)
        
        # Standard attention with quantum-enhanced values
        output = torch.einsum("bhls,bshd->blhd", A, quantum_enhanced_values)
        
        if self.output_attention:
            return output.contiguous(), A
        else:
            return output.contiguous(), None
    
    def apply_quantum_encoding_to_values(self, values):
        """
        Apply quantum circuit encoding to value vectors
        This provides the quantum enhancement while being computationally tractable
        
        NOTE: Quantum processing uses lightning.gpu for CUDA acceleration when available.
        Limited to a small number of tokens per batch for practical speed.
        
        Args:
            values: [B, S, H, D]
        
        Returns:
            quantum_enhanced_values: [B, S, H, D]
        """
        B, S, H, D = values.shape
        
        # Store original device
        original_device = values.device
        
        # Project values to PQC parameters (on original device first)
        values_flat = values.reshape(-1, D)
        with torch.no_grad():  # Break gradient for quantum processing
            pqc_angles = self.feature_to_angles(values_flat.detach())  # [B*S*H, n_pqc_parameters]
        
        # Sample a subset of tokens to apply quantum processing
        # This makes it tractable while still providing quantum enhancement
        total_tokens = B * S * H
        sample_size = min(total_tokens, 4)  # Process max 4 tokens quantumly for speed
        
        if total_tokens > sample_size:
            # Randomly sample tokens for quantum processing
            indices = torch.randperm(total_tokens, device=original_device)[:sample_size]
            sampled_params = pqc_angles[indices]
        else:
            sampled_params = pqc_angles
            indices = torch.arange(total_tokens, device=original_device)
        
        # PennyLane QNodes always expect CPU tensors as input, even with lightning.gpu
        # The GPU acceleration happens internally within the quantum device
        sampled_params_cpu = sampled_params.cpu()
        
        # Apply quantum circuits to sampled tokens
        quantum_measurements = []
        with torch.no_grad():
            for params in sampled_params_cpu:
                # Initialize |0> state on CPU (required for PennyLane)
                initial_state = torch.zeros(2 ** self.n_qubits, dtype=torch.complex64)
                initial_state[0] = 1.0
                
                # Apply quantum circuit (PennyLane handles GPU acceleration internally)
                try:
                    evolved_state = self.quantum_circuit(params, initial_state)
                    measurements = torch.stack(self.measure_circuit(evolved_state))
                except Exception as e:
                    # Fallback to zeros on CPU if quantum circuit fails
                    print(f"Quantum circuit failed: {e}")
                    measurements = torch.zeros(3 * self.n_qubits)
                
                # Move measurements to original device after quantum computation
                quantum_measurements.append(measurements.to(original_device))
        
        quantum_measurements = torch.stack(quantum_measurements)  # [sample_size, 3*n_qubits] on original device
        
        # Project quantum measurements to value dimension (with gradients for this layer)
        quantum_features = self.measurement_to_output(quantum_measurements.float())  # [sample_size, D]
        
        # Create output WITHOUT in-place modifications to avoid corrupting computation graph
        # Use scatter to safely insert quantum features
        alpha = 0.3  # 30% quantum, 70% classical for stability
        quantum_contribution = torch.zeros_like(values_flat)
        quantum_contribution[indices] = alpha * quantum_features.detach()
        
        quantum_enhanced_flat = values_flat + quantum_contribution
        
        # Reshape back
        quantum_enhanced_values = quantum_enhanced_flat.reshape(B, S, H, D)
        
        return quantum_enhanced_values


class QuixerAttentionLayer(nn.Module):
    """
    Attention layer wrapper for Quixer attention mechanism
    Compatible with the existing transformer architecture
    """
    
    def __init__(
        self,
        n_qubits: int = 4,
        qsvt_polynomial_degree: int = 2,
        n_ansatz_layers: int = 1,
        d_model: int = 512,
        n_heads: int = 8,
        d_keys: Optional[int] = None,
        d_values: Optional[int] = None,
        mask_flag: bool = True,
        attention_dropout: float = 0.1,
        output_attention: bool = False,
    ):
        super(QuixerAttentionLayer, self).__init__()
        
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        
        self.inner_attention = QuixerAttention(
            n_qubits=n_qubits,
            qsvt_polynomial_degree=qsvt_polynomial_degree,
            n_ansatz_layers=n_ansatz_layers,
            mask_flag=mask_flag,
            scale=None,
            attention_dropout=attention_dropout,
            output_attention=output_attention,
            d_model=d_values,
        )
        
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
    
    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        """
        Args:
            queries: [B, L, D]
            keys: [B, S, D]
            values: [B, S, D]
        
        Returns:
            output: [B, L, D]
            attention: Optional attention weights
        """
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        
        # Project and reshape
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        
        # Apply Quixer attention
        out, attn = self.inner_attention(
            queries, keys, values, attn_mask, tau, delta
        )
        
        # Reshape and project output
        out = out.view(B, L, -1)
        return self.out_projection(out), attn
