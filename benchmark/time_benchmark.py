import time

import torch
from braket.devices import LocalSimulator


def get_quantum_transformer_time(
    circuit_builder, builder_kwargs, shots=1000, device_mode="local"
):
    """
    Get execution time for quantum transformer circuit.

    Args:
        circuit_builder: Function that builds Braket circuit
        builder_kwargs: Arguments for circuit builder
        shots: Number of measurement shots
        device_mode: "local", "sv1", "tn1", etc.

    Returns:
        dict: Execution time and metadata
    """
    from benchmark import get_device  # Import from your benchmark file

    try:
        # Build circuit
        circuit = circuit_builder(**builder_kwargs)

        # Get device
        device = get_device(device_mode)

        # Time execution
        start_time = time.time()
        task = device.run(circuit, shots=shots)
        result = task.result()
        end_time = time.time()

        execution_time = end_time - start_time

        # Get circuit info for comparison
        from benchmark import resource_counts_from_circuit

        resources = resource_counts_from_circuit(circuit)

        return {
            "execution_time_seconds": execution_time,
            "shots": shots,
            "effective_samples": shots,  # Adjust if postselection used
            "throughput_samples_per_second": shots / execution_time,
            "circuit_qubits": resources["n_qubits"],
            "circuit_gates": resources["one_q_gates"] + resources["two_q_gates_est"],
            "device_mode": device_mode,
        }

    except Exception as e:
        print(f"Quantum execution failed: {e}")
        return None


def get_classical_transformer_time(model, input_tensor, model_type="transformer_layer"):
    """
    Get execution time for classical transformer component.

    Args:
        model: PyTorch model (SimpleAttentionHead or MiniTransformerLayer)
        input_tensor: Input tensor with shape [batch_size, seq_len, d_model]
        model_type: "attention_head" or "transformer_layer"

    Returns:
        dict: Execution time and metadata
    """
    try:
        # Ensure model is in eval mode and on right device
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
            input_tensor = input_tensor.cuda()

        # Warm-up run (exclude from timing)
        with torch.no_grad():
            _ = model(input_tensor)

        if torch.cuda.is_available():
            torch.cuda.synchronize()  # Wait for GPU to finish

        # Timed execution
        start_time = time.time()
        with torch.no_grad():
            output = model(input_tensor)
        end_time = time.time()

        execution_time = end_time - start_time

        # Calculate throughput
        batch_size, seq_len, d_model = input_tensor.shape
        tokens_processed = batch_size * seq_len
        throughput = tokens_processed / execution_time

        return {
            "execution_time_seconds": execution_time,
            "tokens_processed": tokens_processed,
            "throughput_tokens_per_second": throughput,
            "batch_size": batch_size,
            "sequence_length": seq_len,
            "model_parameters": sum(p.numel() for p in model.parameters()),
            "model_type": model_type,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        }

    except Exception as e:
        print(f"Classical execution failed: {e}")
        return None


# Example usage and comparison function
def compare_quantum_classical_performance():
    """Compare quantum and classical transformer performance"""

    # Quantum parameters (matching your Quixer circuits)
    from quixer_benchmark import build_quixer_mini_lcu

    quantum_kwargs = {
        "token0_angles": {"a0": 0.3, "a1": -0.2, "b0": 0.5, "b1": -0.1},
        "token1_angles": {"a0": -0.4, "a1": 0.6, "b0": -0.7, "b1": 0.2},
        "gamma": 0.7,
        "encode_angles": {"x0": 0.2, "x1": -0.1},
        "measure_all": True,
    }

    # Classical parameters (equivalent scale)
    d_model = 96  # Match Quixer's 96 parameters
    batch_size = 1
    seq_len = 2  # Match Quixer's 2 tokens

    # Create classical input
    classical_input = torch.randn(batch_size, seq_len, d_model)

    # Create classical models
    attention_head = SimpleAttentionHead(d_model)
    transformer_layer = MiniTransformerLayer(d_model, num_heads=8)

    print("🔬 Quantum vs Classical Transformer Performance Comparison")
    print("=" * 60)

    # Test quantum
    print("\n🧪 QUANTUM TRANSFORMER (LCU-only):")
    quantum_result = get_quantum_transformer_time(
        build_quixer_mini_lcu, quantum_kwargs, shots=1000, device_mode="local"
    )

    if quantum_result:
        print(f"   Execution time: {quantum_result['execution_time_seconds']:.4f}s")
        print(
            f"   Throughput: {quantum_result['throughput_samples_per_second']:.0f} samples/sec"
        )
        print(
            f"   Circuit: {quantum_result['circuit_qubits']} qubits, {quantum_result['circuit_gates']} gates"
        )

    # Test classical attention head (closest equivalent to LCU)
    print("\n🖥️  CLASSICAL ATTENTION HEAD:")
    classical_attn_result = get_classical_transformer_time(
        attention_head, classical_input, model_type="attention_head"
    )

    if classical_attn_result:
        print(
            f"   Execution time: {classical_attn_result['execution_time_seconds']:.6f}s"
        )
        print(
            f"   Throughput: {classical_attn_result['throughput_tokens_per_second']:.0f} tokens/sec"
        )
        print(f"   Parameters: {classical_attn_result['model_parameters']:,}")

    # Test classical transformer layer (closest equivalent to QSVT circuits)
    print("\n🖥️  CLASSICAL TRANSFORMER LAYER:")
    classical_transformer_result = get_classical_transformer_time(
        transformer_layer, classical_input, model_type="transformer_layer"
    )

    if classical_transformer_result:
        print(
            f"   Execution time: {classical_transformer_result['execution_time_seconds']:.6f}s"
        )
        print(
            f"   Throughput: {classical_transformer_result['throughput_tokens_per_second']:.0f} tokens/sec"
        )
        print(f"   Parameters: {classical_transformer_result['model_parameters']:,}")

    # Comparative analysis
    if quantum_result and classical_attn_result:
        print("\n📊 COMPARATIVE ANALYSIS:")
        print(
            f"   Time ratio (Classical/Quantum): {classical_attn_result['execution_time_seconds'] / quantum_result['execution_time_seconds']:.2f}x"
        )
        print(
            f"   Throughput ratio: {classical_attn_result['throughput_tokens_per_second'] / quantum_result['throughput_samples_per_second']:.2f}x"
        )

        # Normalized comparison
        quantum_time_per_param = (
            quantum_result["execution_time_seconds"] / 96
        )  # Quixer has ~96 params
        classical_time_per_param = (
            classical_attn_result["execution_time_seconds"]
            / classical_attn_result["model_parameters"]
        )
        print(f"   Time per parameter (quantum): {quantum_time_per_param:.6f}s")
        print(f"   Time per parameter (classical): {classical_time_per_param:.6f}s")
        print(
            f"   Efficiency ratio: {classical_time_per_param / quantum_time_per_param:.2f}x"
        )


if __name__ == "__main__":
    compare_quantum_classical_performance()
