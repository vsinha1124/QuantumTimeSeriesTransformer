# classical_benchmark.py
import time
import torch
import torch.nn as nn
import numpy as np

class ClassicalArchitectureBenchmark:
    """Benchmark classical architectures equivalent to quantum ones"""

    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        self.architectures = {}

    class LCUEquivalent(nn.Module):
        """Classical equivalent to Quantum LCU_Linear: Linear token mixing"""
        def __init__(self, d_model=64, seq_len=2):
            super().__init__()
            self.d_model = d_model
            self.seq_len = seq_len
            # Equivalent to quantum LCU: linear combination of representations
            self.mixing_weights = nn.Parameter(torch.randn(seq_len, seq_len))
            # Additional linear transformation
            self.linear = nn.Linear(d_model, d_model, bias=False)

        def forward(self, x):
            # x: [batch_size, seq_len, d_model]
            # Linear mixing along sequence dimension (equivalent to LCU)
            mixing_matrix = self.mixing_weights.softmax(dim=-1)
            mixed = torch.einsum('mn,bnd->bmd', mixing_matrix, x)
            # Additional linear transform
            output = self.linear(mixed)
            return output

    class QSVT_SingleEquivalent(nn.Module):
        """Classical equivalent to Quantum QSVT_Single: Nonlinear feature transform"""
        def __init__(self, d_model=64):
            super().__init__()
            self.d_model = d_model
            # Equivalent to QSVT polynomial transformation
            self.transform = nn.Sequential(
                nn.Linear(d_model, d_model * 2),  # Expand
                nn.GELU(),                        # Nonlinearity
                nn.Linear(d_model * 2, d_model),  # Contract
                nn.LayerNorm(d_model)             # Normalization
            )

        def forward(self, x):
            # x: [batch_size, seq_len, d_model]
            # Apply nonlinear transformation to each token
            return self.transform(x)

    class QSVT_FullEquivalent(nn.Module):
        """Classical equivalent to Quantum QSVT_Full: Complete attention mechanism"""
        def __init__(self, d_model=64, seq_len=2, num_heads=4):
            super().__init__()
            self.d_model = d_model
            self.seq_len = seq_len

            # Multi-head attention (equivalent to full QSVT on LCU)
            self.attention = nn.MultiheadAttention(
                d_model, num_heads, batch_first=True, dropout=0.1
            )

            # Feed-forward network with nonlinearities
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Linear(d_model * 4, d_model),
                nn.Dropout(0.1)
            )

            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)

        def forward(self, x):
            # Self-attention (equivalent to QSVT nonlinear mixing)
            attn_out, _ = self.attention(x, x, x)
            x = self.norm1(x + attn_out)

            # Feed-forward (additional nonlinear transformation)
            ff_out = self.ffn(x)
            x = self.norm2(x + ff_out)

            return x

    def benchmark_classical_architectures(self, batch_size=32, seq_len=2, d_model=64, repeats=100):
        """Benchmark classical architectures equivalent to quantum ones"""
        print("=== Classical Architecture Benchmarks (Quantum-Equivalent) ===")

        architectures = {
            "Classical_LCU_Equivalent": {
                "module": self.LCUEquivalent(d_model, seq_len),
                "description": "Linear token mixing (equivalent to Quantum LCU)",
                "input_shape": (batch_size, seq_len, d_model)
            },

            "Classical_QSVT_Single_Equivalent": {
                "module": self.QSVT_SingleEquivalent(d_model),
                "description": "Nonlinear feature transform (equivalent to Quantum QSVT Single)",
                "input_shape": (batch_size, seq_len, d_model)
            },

            "Classical_QSVT_Full_Equivalent": {
                "module": self.QSVT_FullEquivalent(d_model, seq_len),
                "description": "Complete attention mechanism (equivalent to Quantum QSVT Full)",
                "input_shape": (batch_size, seq_len, d_model)
            }
        }

        for name, arch in architectures.items():
            print(f"\n--- Benchmarking {name} ---")
            print(f"Description: {arch['description']}")

            model = arch["module"].to(self.device)
            x = torch.randn(arch["input_shape"]).to(self.device)

            # Warmup
            for _ in range(10):
                _ = model(x)

            # Time forward pass
            times = []
            for _ in range(repeats):
                start_time = time.time()
                output = model(x)
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                end_time = time.time()
                times.append(end_time - start_time)

            avg_time = np.mean(times)
            std_time = np.std(times)

            # Calculate throughput (samples per second)
            throughput = batch_size / avg_time if avg_time > 0 else float('inf')

            # Parameter count
            param_count = sum(p.numel() for p in model.parameters())

            # Memory usage (approximate)
            if self.device.type == 'cpu':
                import psutil
                import os
                process = psutil.Process(os.getpid())
                memory_usage = process.memory_info().rss / 1024 / 1024  # MB
            else:
                memory_usage = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB

            self.architectures[name] = {
                "description": arch["description"],
                "execution_time_s": avg_time,
                "execution_time_std_s": std_time,
                "throughput_samples_s": throughput,
                "parameters": param_count,
                "memory_mb": memory_usage,
                "output_shape": tuple(output.shape)
            }

            self._print_architecture_results(name, self.architectures[name])

        return self.architectures

    def _print_architecture_results(self, name, results):
        """Print formatted results for a classical architecture"""
        print(f"Architecture: {name}")
        print(f"  Description: {results['description']}")
        print(f"  Parameters: {results['parameters']:,}")
        print(f"  Execution Time: {results['execution_time_s']:.6f}s ± {results['execution_time_std_s']:.6f}s")
        print(f"  Throughput: {results['throughput_samples_s']:.0f} samples/s")
        print(f"  Memory Usage: {results['memory_mb']:.1f} MB")
        print(f"  Output Shape: {results['output_shape']}")


    # classical_benchmark.py (add this method to the existing class)
    def export_classical_results_to_csv(self, filename="classical_results.csv"):
        """Export classical results to CSV for cross-environment comparison"""
        import csv
        import os
        import time

        fieldnames = [
            'architecture', 'description', 'execution_time_s', 'throughput_samples_s',
            'parameters', 'memory_mb', 'output_shape', 'environment', 'timestamp'
        ]

        file_exists = os.path.isfile(filename)

        with open(filename, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            for name, results in self.architectures.items():
                writer.writerow({
                    'architecture': name,
                    'description': results['description'],
                    'execution_time_s': results['execution_time_s'],
                    'throughput_samples_s': results['throughput_samples_s'],
                    'parameters': results['parameters'],
                    'memory_mb': results['memory_mb'],
                    'output_shape': str(results['output_shape']),
                    'environment': 'colab_classical',
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                })

        print(f"Classical results exported to {filename}")

    def generate_classical_report(self):
        """Generate classical-only performance report"""
        print("\n" + "="*60)
        print("CLASSICAL ARCHITECTURE BENCHMARK REPORT")
        print("="*60)

        print(f"{'Architecture':<35} {'Parameters':<12} {'Time (s)':<12} {'Throughput':<12} {'Memory (MB)':<10}")
        print("-" * 85)

        for name, results in self.architectures.items():
            print(f"{name:<35} {results['parameters']:<12,} {results['execution_time_s']:<12.6f} "
                  f"{results['throughput_samples_s']:<12.0f} {results['memory_mb']:<10.1f}")

        # Architecture comparison insights
        print("\n--- Classical Architecture Insights ---")
        fastest = min(self.architectures.items(), key=lambda x: x[1]['execution_time_s'])
        most_efficient = min(self.architectures.items(), key=lambda x: x[1]['parameters'])
        highest_throughput = max(self.architectures.items(), key=lambda x: x[1]['throughput_samples_s'])

        print(f"Fastest: {fastest[0]} ({fastest[1]['execution_time_s']:.6f}s)")
        print(f"Most Parameter-Efficient: {most_efficient[0]} ({most_efficient[1]['parameters']:,} parameters)")
        print(f"Highest Throughput: {highest_throughput[0]} ({highest_throughput[1]['throughput_samples_s']:.0f} samples/s)")

if __name__ == "__main__":
    # Run classical benchmarks
    classical_benchmark = ClassicalArchitectureBenchmark(device="cuda")  # or "cpu"
    classical_results = classical_benchmark.benchmark_classical_architectures(
        batch_size=32,
        seq_len=2,  # Match quantum token count
        d_model=64,
        repeats=100
    )
    classical_benchmark.generate_classical_report()
    classical_benchmark.export_classical_results_to_csv("classical_results.csv")
