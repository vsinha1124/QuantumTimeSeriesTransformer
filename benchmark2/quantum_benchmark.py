# quantum_transformer_benchmark.py
import time
import numpy as np
import torch
from braket.circuits import Circuit
from benchmark import benchmark_circuit_builder, get_device
from quixer_benchmark import (
    build_quixer_mini_lcu,
    build_quixer_mini_with_qsvt_U,
    build_quixer_mini_with_qsvt_full_lcu
)

class QuantumTransformerBenchmark:
    """Benchmark suite for quantum transformer components"""

    def __init__(self, device_mode="local"):
        self.device_mode = device_mode
        self.results = {}

    def benchmark_quixer_variants(self, shots=1000, repeats=3):
        """Benchmark different Quixer architectural variants"""
        print("=== Quantum Transformer Benchmarks ===")

        # Standard parameters for all variants
        base_params = {
            "token0_angles": {"a0": 0.3, "a1": -0.2, "b0": 0.5, "b1": -0.1},
            "token1_angles": {"a0": -0.4, "a1": 0.6, "b0": -0.7, "b1": 0.2},
            "gamma": 0.7,
            "encode_angles": {"x0": 0.2, "x1": -0.1},
            "qsvt_phis": (0.3, 0.9, -0.4),
            "measure_all": True
        }

        variants = [
            ("LCU_Linear", build_quixer_mini_lcu, {
                "token0_angles": base_params["token0_angles"],
                "token1_angles": base_params["token1_angles"],
                "gamma": base_params["gamma"],
                "encode_angles": base_params["encode_angles"],
                "measure_all": base_params["measure_all"]
            }),

            ("QSVT_SingleUnitary", build_quixer_mini_with_qsvt_U, {
                "token_angles": base_params["token0_angles"],
                "encode_angles": base_params["encode_angles"],
                "qsvt_phis": base_params["qsvt_phis"],
                "measure_all": base_params["measure_all"]
            }),

            ("QSVT_FullLCU", build_quixer_mini_with_qsvt_full_lcu, {
                "token0_angles": base_params["token0_angles"],
                "token1_angles": base_params["token1_angles"],
                "gamma": base_params["gamma"],
                "encode_angles": base_params["encode_angles"],
                "qsvt_phis": base_params["qsvt_phis"],
                "measure_all": base_params["measure_all"]
            })
        ]

        for name, builder, kwargs in variants:
            print(f"\n--- Benchmarking {name} ---")

            summary, detailed_results = benchmark_circuit_builder(
                builder,
                kwargs,
                shots=shots,
                repeats=repeats,
                postselect_bit=None,  # Auto-detect
                csv_out=f"quantum_{name.lower()}_benchmark.csv"
            )

            self.results[name] = {
                "summary": summary,
                "detailed": detailed_results,
                "resources": detailed_results[0] if detailed_results else {}
            }

            self._print_variant_summary(name, summary, detailed_results[0] if detailed_results else {})

    def benchmark_scaling(self, max_qubits=8, shots=500):
        """Benchmark scaling with increasing qubit count"""
        print("\n=== Quantum Scaling Analysis ===")

        scaling_results = {}

        for n_qubits in range(2, max_qubits + 1, 2):
            print(f"\n--- Testing {n_qubits} qubits ---")

            # Create a simple circuit that scales with qubit count
            circ = Circuit()

            # Add encoding on all qubits
            for q in range(n_qubits):
                circ.ry(q, 0.5 * q)

            # Add entangling layers (scales with qubit count)
            for q in range(n_qubits - 1):
                circ.cnot(q, q + 1)

            # Add parameterized rotations
            for q in range(n_qubits):
                circ.rz(q, 0.3 * q)

            # Measure all qubits
            for q in range(n_qubits):
                circ.measure(q)

            # Time the execution
            device = get_device()
            start_time = time.time()

            try:
                task = device.run(circ, shots=shots)
                result = task.result()
                execution_time = time.time() - start_time

                scaling_results[n_qubits] = {
                    "execution_time": execution_time,
                    "qubit_count": n_qubits,
                    "gate_count": len(circ.instructions),
                    "success": True
                }

                print(f"  Qubits: {n_qubits}, Time: {execution_time:.3f}s, Gates: {len(circ.instructions)}")

            except Exception as e:
                scaling_results[n_qubits] = {
                    "execution_time": float('inf'),
                    "qubit_count": n_qubits,
                    "gate_count": len(circ.instructions),
                    "success": False,
                    "error": str(e)
                }
                print(f"  Qubits: {n_qubits}, Failed: {e}")

        self.results["scaling"] = scaling_results
        return scaling_results

    def _print_variant_summary(self, name, summary, resources):
        """Print formatted summary for a quantum variant"""
        print(f"Architecture: {name}")
        print(f"  Qubits: {resources.get('n_qubits', 'N/A')}")
        print(f"  1Q Gates: {resources.get('one_q_gates', 'N/A')}")
        print(f"  2Q Gates: {resources.get('two_q_gates_est', 'N/A')}")
        print(f"  Success Probability: {summary['mean_postselect_prob']:.4f} ± {summary['std_postselect_prob']:.4f}")
        print(f"  Execution Time: {summary['mean_wall_time_s']:.3f}s ± {summary['std_wall_time_s']:.3f}s")
        print(f"  Throughput: {summary['mean_throughput']:.1f} ± {summary['std_throughput']:.1f} samples/s")

        if 'z0_mean' in resources:
            print(f"  ⟨Z₀⟩: {resources['z0_mean']:.4f} ± {resources['z0_sem']:.4f}")

    def generate_report(self):
        """Generate comprehensive benchmark report"""
        print("\n" + "="*60)
        print("QUANTUM TRANSFORMER BENCHMARK REPORT")
        print("="*60)

        # Architecture comparison
        if any(k in self.results for k in ["LCU_Linear", "QSVT_SingleUnitary", "QSVT_FullLCU"]):
            print("\n--- Architecture Comparison ---")
            headers = ["Variant", "Qubits", "1Q Gates", "2Q Gates", "Success Prob", "Time (s)", "Throughput"]
            print(f"{headers[0]:<15} {headers[1]:<6} {headers[2]:<8} {headers[3]:<8} {headers[4]:<12} {headers[5]:<8} {headers[6]:<10}")
            print("-" * 80)

            for name in ["LCU_Linear", "QSVT_SingleUnitary", "QSVT_FullLCU"]:
                if name in self.results:
                    res = self.results[name]
                    resources = res["resources"]
                    summary = res["summary"]

                    print(f"{name:<15} {resources.get('n_qubits', 'N/A'):<6} "
                          f"{resources.get('one_q_gates', 'N/A'):<8} "
                          f"{resources.get('two_q_gates_est', 'N/A'):<8} "
                          f"{summary['mean_postselect_prob']:<12.4f} "
                          f"{summary['mean_wall_time_s']:<8.3f} "
                          f"{summary['mean_throughput']:<10.1f}")

        # Scaling analysis
        if "scaling" in self.results:
            print("\n--- Scaling Analysis ---")
            print("Qubits | Time (s) | Gates | Status")
            print("-" * 40)
            for n_qubits, result in sorted(self.results["scaling"].items()):
                status = "✓" if result["success"] else "✗"
                print(f"{n_qubits:6} | {result['execution_time']:8.3f} | "
                      f"{result['gate_count']:5} | {status}")

if __name__ == "__main__":
    # Run quantum benchmarks
    quantum_benchmark = QuantumTransformerBenchmark(device_mode="local")

    # Benchmark different architectural variants
    quantum_benchmark.benchmark_quixer_variants(shots=1000, repeats=3)

    # Benchmark scaling behavior
    quantum_benchmark.benchmark_scaling(max_qubits=6, shots=500)

    # Generate final report
    quantum_benchmark.generate_report()
