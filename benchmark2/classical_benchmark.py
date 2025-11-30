# classical_benchmark.py
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.benchmark import Timer, Compare
import psutil
import os

class ClassicalTransformerBenchmark:
    """Benchmark suite for classical transformer components"""

    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        self.results = {}

    class SimpleAttentionHead(nn.Module):
        """Simplified attention head similar to Quixer's token mixing"""
        def __init__(self, d_model):
            super().__init__()
            self.d_model = d_model
            self.w_q = nn.Linear(d_model, d_model, bias=False)
            self.w_k = nn.Linear(d_model, d_model, bias=False)
            self.w_v = nn.Linear(d_model, d_model, bias=False)

        def forward(self, x):
            Q = self.w_q(x)
            K = self.w_k(x)
            V = self.w_v(x)

            attn_weights = torch.softmax(
                Q @ K.transpose(-2, -1) / np.sqrt(self.d_model), dim=-1
            )
            output = attn_weights @ V
            return output

    class MiniTransformerLayer(nn.Module):
        """Mini transformer layer for fair comparison with quantum version"""
        def __init__(self, d_model, num_heads):
            super().__init__()
            self.attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
            self.ffn = nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.GELU(),
                nn.Linear(4 * d_model, d_model)
            )
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)

        def forward(self, x):
            # Self-attention
            attn_out, _ = self.attention(x, x, x)
            x = self.norm1(x + attn_out)

            # Feed-forward
            ff_out = self.ffn(x)
            x = self.norm2(x + ff_out)
            return x

    class LinearMixingLayer(nn.Module):
        """Linear mixing layer equivalent to Quixer's LCU approach"""
        def __init__(self, d_model, seq_len):
            super().__init__()
            self.mixing_weights = nn.Parameter(torch.randn(seq_len, seq_len))
            self.d_model = d_model
            self.seq_len = seq_len

        def forward(self, x):
            # Linear mixing along sequence dimension
            # x shape: [batch_size, seq_len, d_model]
            # mixing_weights shape: [seq_len, seq_len]
            mixing_matrix = self.mixing_weights.softmax(dim=-1)

            # Correct einsum: mix sequence dimensions, preserve batch and feature dimensions
            return torch.einsum('mn,bnd->bmd', mixing_matrix, x)

    class FourierMixingLayer(nn.Module):
        """Fourier mixing layer similar to FNet architecture"""
        def __init__(self, d_model):
            super().__init__()
            self.d_model = d_model

        def forward(self, x):
            # Apply FFT along sequence dimension and take real part
            # This mimics FNet's approach mentioned in the Quixer paper
            x_fft = torch.fft.fft(x, dim=1)
            return x_fft.real

    def benchmark_components(self, batch_size=32, seq_len=32, d_model=64, repeats=100):
        """Benchmark different classical transformer components"""
        print("=== Classical Transformer Benchmarks ===")

        # Test configurations matching quantum scale
        configs = [
            ("SimpleAttention", {"d_model": d_model}),
            ("LinearMixing", {"d_model": d_model, "seq_len": seq_len}),
            ("FourierMixing", {"d_model": d_model}),
            ("MiniTransformer", {"d_model": d_model, "num_heads": 8})
        ]

        for name, params in configs:
            print(f"\n--- Benchmarking {name} ---")

            try:
                # Create model and test input
                if name == "SimpleAttention":
                    model = self.SimpleAttentionHead(params["d_model"]).to(self.device)
                elif name == "LinearMixing":
                    model = self.LinearMixingLayer(params["d_model"], params["seq_len"]).to(self.device)
                elif name == "FourierMixing":
                    model = self.FourierMixingLayer(params["d_model"]).to(self.device)
                else:  # MiniTransformer
                    model = self.MiniTransformerLayer(params["d_model"], params["num_heads"]).to(self.device)

                x = torch.randn(batch_size, seq_len, params["d_model"]).to(self.device)

                # Warmup
                for _ in range(10):
                    _ = model(x)

                # Time forward pass
                start_time = time.time()
                for _ in range(repeats):
                    output = model(x)
                    if self.device.type == 'cuda':
                        torch.cuda.synchronize()
                forward_time = (time.time() - start_time) / repeats

                # Memory usage (approximate)
                if self.device.type == 'cpu':
                    process = psutil.Process(os.getpid())
                    memory_usage = process.memory_info().rss / 1024 / 1024  # MB
                else:
                    memory_usage = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
                    torch.cuda.reset_peak_memory_stats()

                # Parameter count
                param_count = sum(p.numel() for p in model.parameters())

                # Throughput calculation
                throughput = batch_size * seq_len / forward_time if forward_time > 0 else float('inf')

                self.results[name] = {
                    "forward_time_ms": forward_time * 1000,
                    "memory_mb": memory_usage,
                    "parameters": param_count,
                    "throughput_tokens_s": throughput,
                    "output_shape": tuple(output.shape)
                }

                self._print_component_summary(name, self.results[name])

            except Exception as e:
                print(f"Error benchmarking {name}: {e}")
                self.results[name] = {
                    "forward_time_ms": float('inf'),
                    "memory_mb": 0,
                    "parameters": 0,
                    "throughput_tokens_s": 0,
                    "output_shape": None,
                    "error": str(e)
                }

    def benchmark_scaling(self, max_dim=256, batch_size=8, seq_len=32):
        """Benchmark scaling with increasing model dimensions"""
        print("\n=== Classical Scaling Analysis ===")

        scaling_results = {}
        dimensions = [64, 128, 192, 256]  # Match quantum state space scaling

        for d_model in dimensions:
            if d_model > max_dim:
                continue

            print(f"\n--- Testing d_model={d_model} ---")

            try:
                model = self.MiniTransformerLayer(d_model, num_heads=8).to(self.device)
                x = torch.randn(batch_size, seq_len, d_model).to(self.device)

                # Warmup
                for _ in range(5):
                    _ = model(x)

                # Time execution
                start_time = time.time()
                for _ in range(50):
                    output = model(x)
                    if self.device.type == 'cuda':
                        torch.cuda.synchronize()
                execution_time = (time.time() - start_time) / 50

                # Resource metrics
                param_count = sum(p.numel() for p in model.parameters())
                flops_estimate = self._estimate_flops(model, x)

                scaling_results[d_model] = {
                    "execution_time_ms": execution_time * 1000,
                    "parameters": param_count,
                    "flops_estimate": flops_estimate,
                    "throughput_tokens_s": (batch_size * seq_len) / execution_time,
                    "success": True
                }

                print(f"  d_model: {d_model}, Time: {execution_time*1000:.2f}ms, "
                      f"Params: {param_count:,}, FLOPS: {flops_estimate:.2e}")

            except Exception as e:
                print(f"  d_model: {d_model}, Failed: {e}")
                scaling_results[d_model] = {
                    "execution_time_ms": float('inf'),
                    "parameters": 0,
                    "flops_estimate": 0,
                    "throughput_tokens_s": 0,
                    "success": False,
                    "error": str(e)
                }

        self.results["scaling"] = scaling_results
        return scaling_results

    def _estimate_flops(self, model, x):
        """Rough FLOPs estimation for transformer components"""
        batch_size, seq_len, d_model = x.shape

        # Attention FLOPs: ~4 * batch_size * seq_len^2 * d_model
        attention_flops = 4 * batch_size * (seq_len ** 2) * d_model

        # FFN FLOPs: ~2 * batch_size * seq_len * d_model * (4 * d_model)
        ffn_flops = 2 * batch_size * seq_len * d_model * (4 * d_model)

        # Layer norm FLOPs: ~2 * batch_size * seq_len * d_model
        norm_flops = 2 * batch_size * seq_len * d_model

        total_flops = attention_flops + ffn_flops + 2 * norm_flops  # Two layer norms
        return total_flops

    def _print_component_summary(self, name, results):
        """Print formatted summary for a classical component"""
        print(f"Component: {name}")
        print(f"  Forward Time: {results['forward_time_ms']:.2f} ms")
        print(f"  Memory Usage: {results['memory_mb']:.1f} MB")
        print(f"  Parameters: {results['parameters']:,}")
        print(f"  Throughput: {results['throughput_tokens_s']:.0f} tokens/s")
        print(f"  Output Shape: {results['output_shape']}")

    def benchmark_equivalent_quantum_scale(self):
        """Benchmark at scales equivalent to quantum model (6 qubits = 64 complex dimensions)"""
        print("\n=== Quantum-Equivalent Scale Benchmarks ===")

        # 6 qubits → 2^6 = 64 complex dimensions ≈ 128 real parameters
        quantum_equivalent_dims = [64, 128]
        seq_lengths = [8, 16, 32]  # Context lengths

        equivalent_results = {}

        for d_model in quantum_equivalent_dims:
            for seq_len in seq_lengths:
                key = f"d{d_model}_seq{seq_len}"
                print(f"\n--- Testing {key} ---")

                try:
                    model = self.MiniTransformerLayer(d_model, num_heads=8).to(self.device)
                    x = torch.randn(1, seq_len, d_model).to(self.device)  # batch_size=1 for fair comparison

                    # Time execution
                    start_time = time.time()
                    for _ in range(100):
                        output = model(x)
                        if self.device.type == 'cuda':
                            torch.cuda.synchronize()
                    execution_time = (time.time() - start_time) / 100

                    equivalent_results[key] = {
                        "execution_time_ms": execution_time * 1000,
                        "parameters": sum(p.numel() for p in model.parameters()),
                        "throughput_tokens_s": seq_len / execution_time,
                        "success": True
                    }

                    print(f"  Time: {execution_time*1000:.2f}ms, "
                          f"Params: {equivalent_results[key]['parameters']:,}, "
                          f"Throughput: {equivalent_results[key]['throughput_tokens_s']:.1f} tokens/s")

                except Exception as e:
                    print(f"  Configuration {key} failed: {e}")
                    equivalent_results[key] = {
                        "execution_time_ms": float('inf'),
                        "parameters": 0,
                        "throughput_tokens_s": 0,
                        "success": False,
                        "error": str(e)
                    }

        self.results["quantum_equivalent"] = equivalent_results
        return equivalent_results

    def generate_report(self):
        """Generate comprehensive benchmark report"""
        print("\n" + "="*60)
        print("CLASSICAL TRANSFORMER BENCHMARK REPORT")
        print("="*60)

        # Component comparison
        if self.results:
            print("\n--- Component Comparison ---")
            headers = ["Component", "Time (ms)", "Memory (MB)", "Params", "Throughput (tokens/s)"]
            print(f"{headers[0]:<20} {headers[1]:<10} {headers[2]:<12} {headers[3]:<12} {headers[4]:<15}")
            print("-" * 80)

            for name, results in self.results.items():
                if name not in ["scaling", "quantum_equivalent"] and "error" not in results:
                    print(f"{name:<20} {results['forward_time_ms']:<10.2f} "
                          f"{results['memory_mb']:<12.1f} {results['parameters']:<12,} "
                          f"{results['throughput_tokens_s']:<15.0f}")

        # Scaling analysis
        if "scaling" in self.results:
            print("\n--- Scaling Analysis ---")
            print("d_model | Time (ms) | Parameters | Throughput (tokens/s)")
            print("-" * 60)
            for d_model, result in sorted(self.results["scaling"].items()):
                if result.get("success", False):
                    print(f"{d_model:7} | {result['execution_time_ms']:9.2f} | "
                          f"{result['parameters']:10,} | {result['throughput_tokens_s']:17.0f}")

        # Quantum equivalent comparison
        if "quantum_equivalent" in self.results:
            print("\n--- Quantum-Equivalent Scale ---")
            print("Configuration | Time (ms) | Parameters | Throughput (tokens/s)")
            print("-" * 65)
            for config, result in self.results["quantum_equivalent"].items():
                if result.get("success", False):
                    print(f"{config:12} | {result['execution_time_ms']:9.2f} | "
                          f"{result['parameters']:10,} | {result['throughput_tokens_s']:17.1f}")

if __name__ == "__main__":
    # Detect available device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Run classical benchmarks
    classical_benchmark = ClassicalTransformerBenchmark(device=device)

    # Benchmark different components
    classical_benchmark.benchmark_components(
        batch_size=32,
        seq_len=32,
        d_model=64,  # Equivalent to 6-qubit quantum state
        repeats=100
    )

    # Benchmark scaling behavior
    classical_benchmark.benchmark_scaling(max_dim=256, batch_size=8, seq_len=32)

    # Benchmark at quantum-equivalent scales
    classical_benchmark.benchmark_equivalent_quantum_scale()

    # Generate final report
    classical_benchmark.generate_report()
