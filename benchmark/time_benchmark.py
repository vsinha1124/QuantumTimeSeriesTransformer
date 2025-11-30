import time
import torch

# Uses your unified device selector
from benchmark import get_device, resource_counts_from_circuit
from classical_benchmark import SimpleAttentionHead, MiniTransformerLayer


def get_quantum_transformer_time(
    circuit_builder, builder_kwargs, shots=1000, device_mode="local"
):
    """
    Benchmark execution time for a single quantum circuit variant.
    Uses the unified get_device() from benchmark.py (DEVICE_MODE-controlled).
    """

    # Build circuit
    circuit = circuit_builder(**builder_kwargs)

    # Select device (uses the DEVICE_MODE inside benchmark.py)
    device = get_device()

    start = time.time()
    task = device.run(circuit, shots=shots)
    result = task.result()
    end = time.time()

    flat_counts = result.measurement_counts
    resources = resource_counts_from_circuit(circuit)

    exec_time = end - start

    return {
        "execution_time_seconds": exec_time,
        "shots": shots,
        "effective_samples": shots,  # add postselection later if needed
        "throughput_samples_per_second": shots / exec_time,
        "circuit_qubits": resources["n_qubits"],
        "circuit_gates": resources["one_q_gates"] + resources["two_q_gates_est"],
        "device_mode": device_mode,
    }


def get_classical_transformer_time(model, input_tensor):
    """
    Benchmark a classical transformer component on CPU or GPU.
    """

    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()
        input_tensor = input_tensor.cuda()

    # Warm-up
    with torch.no_grad():
        _ = model(input_tensor)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.time()
    with torch.no_grad():
        out = model(input_tensor)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.time()

    exec_time = end - start

    batch, seq, d_model = input_tensor.shape
    tokens = batch * seq

    return {
        "execution_time_seconds": exec_time,
        "tokens_processed": tokens,
        "throughput_tokens_per_second": tokens / exec_time,
        "batch_size": batch,
        "sequence_length": seq,
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }


# -------- Performance Comparison Wrapper -------- #

def compare_quantum_classical_performance():
    """Compare quantum vs classical forward-pass latency."""

    from quixer_benchmark import build_quixer_mini_lcu

    # Quantum circuit parameters
    quantum_kwargs = {
        "token0_angles": {"a0": 0.3, "a1": -0.2, "b0": 0.5, "b1": -0.1},
        "token1_angles": {"a0": -0.4, "a1": 0.6, "b0": -0.7, "b1": 0.2},
        "gamma": 0.7,
        "encode_angles": {"x0": 0.2, "x1": -0.1},
        "measure_all": True,
    }

    # Classical embedding dimension to roughly match Quixer parameters
    d_model = 96
    batch_size = 1
    seq_len = 2

    classical_input = torch.randn(batch_size, seq_len, d_model)

    attention_head = SimpleAttentionHead(d_model)
    transformer_layer = MiniTransformerLayer(d_model, num_heads=8)

    print("\n🔬 Quantum vs Classical Transformer Performance")
    print("================================================")

    # ------------ Quantum Benchmark ------------
    print("\n🧪 QUANTUM (LCU-only):")
    q_result = get_quantum_transformer_time(
        build_quixer_mini_lcu,
        quantum_kwargs,
        shots=1000,
        device_mode="local",
    )

    if q_result:
        print(f"   Time:       {q_result['execution_time_seconds']:.4f} s")
        print(f"   Throughput: {q_result['throughput_samples_per_second']:.0f} samples/s")
        print(f"   Qubits:     {q_result['circuit_qubits']}, Gates: {q_result['circuit_gates']}")

    # ------------ Classical Attention Head ------------
    print("\n🖥️  CLASSICAL ATTENTION HEAD:")
    c_attn = get_classical_transformer_time(attention_head, classical_input)

    if c_attn:
        print(f"   Time:       {c_attn['execution_time_seconds']:.6f} s")
        print(f"   Throughput: {c_attn['throughput_tokens_per_second']:.0f} tokens/s")
        print(f"   Params:     {c_attn['model_parameters']:,}")

    # ------------ Classical Transformer Layer ------------
    print("\n🖥️  CLASSICAL TRANSFORMER LAYER:")
    c_tf = get_classical_transformer_time(transformer_layer, classical_input)

    if c_tf:
        print(f"   Time:       {c_tf['execution_time_seconds']:.6f} s")
        print(f"   Throughput: {c_tf['throughput_tokens_per_second']:.0f} tokens/s")
        print(f"   Params:     {c_tf['model_parameters']:,}")

    # ------------ Comparison ------------
    print("\n📊 COMPARATIVE RATIOS")
    if q_result and c_attn:
        ratio_time = c_attn["execution_time_seconds"] / q_result["execution_time_seconds"]
        ratio_throughput = (
            c_attn["throughput_tokens_per_second"] /
            q_result["throughput_samples_per_second"]
        )

        print(f"   Classical/Quantum Time:       {ratio_time:.2f}×")
        print(f"   Classical/Quantum Throughput: {ratio_throughput:.2f}×")

    print("\nDone.\n")


if __name__ == "__main__":
    compare_quantum_classical_performance()
