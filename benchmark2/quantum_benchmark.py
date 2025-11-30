# quantum_benchmark.py
import time
import numpy as np
from braket.circuits import Circuit
from benchmark import benchmark_circuit_builder, get_device
# Import the global DEVICE_MODE to modify it
from benchmark import DEVICE_MODE as global_device_mode
from quixer_architecture import (
    build_quixer_mini_lcu,
    build_quixer_mini_with_qsvt_U,
    build_quixer_mini_with_qsvt_full_lcu
)

class QuantumArchitectureBenchmark:
    """Benchmark three distinct quantum transformer architectures"""

    def __init__(self, device_mode="local"):
        self.device_mode = device_mode
        self.architectures = {}
        # Set the global device mode when instance is created
        global global_device_mode
        global_device_mode = device_mode
        print(f"Quantum benchmark configured for device: {device_mode}")

    def benchmark_quantum_architectures(self, shots=1000, repeats=3):
        """Benchmark the three quantum architectures"""
        print("=== Quantum Architecture Benchmarks ===")
        print(f"Using device: {self.device_mode}")

        # Test device connectivity
        try:
            device = get_device()
            print(f"Connected to: {getattr(device, 'name', str(device))}")

            # For real quantum devices, check status
            if hasattr(device, 'status'):
                print(f"Device status: {device.status}")
        except Exception as e:
            print(f"Warning: Could not connect to device: {e}")
            print("Falling back to local simulator")
            global global_device_mode
            global_device_mode = "local"

        # Define the three quantum architectures with their parameters
        architectures = {
            "Quantum_LCU_Linear": {
                "builder": build_quixer_mini_lcu,
                "params": {
                    "token0_angles": {"a0": 0.3, "a1": -0.2, "b0": 0.5, "b1": -0.1},
                    "token1_angles": {"a0": -0.4, "a1": 0.6, "b0": -0.7, "b1": 0.2},
                    "gamma": 0.7,
                    "encode_angles": {"x0": 0.2, "x1": -0.1},
                    "measure_all": True
                },
                "description": "Linear combination of token unitaries (basic mixing)"
            },

            "Quantum_QSVT_Single": {
                "builder": build_quixer_mini_with_qsvt_U,
                "params": {
                    "token_angles": {"a0": 0.3, "a1": 0.5, "b0": -0.2, "b1": 0.4},
                    "encode_angles": {"x0": 0.2, "x1": -0.1},
                    "qsvt_phis": (0.3, 0.9, -0.4),
                    "measure_all": True
                },
                "description": "Nonlinear transform on single unitary (feature transformation)"
            },

            "Quantum_QSVT_Full": {
                "builder": build_quixer_mini_with_qsvt_full_lcu,
                "params": {
                    "token0_angles": {"a0": 0.3, "a1": -0.2, "b0": 0.5, "b1": -0.1},
                    "token1_angles": {"a0": -0.4, "a1": 0.6, "b0": -0.7, "b1": 0.2},
                    "gamma": 0.7,
                    "encode_angles": {"x0": 0.2, "x1": -0.1},
                    "qsvt_phis": (0.3, 0.9, -0.4),
                    "measure_all": True
                },
                "description": "Full nonlinear transform on mixed unitaries (complete attention)"
            }
        }

        for name, arch in architectures.items():
            print(f"\n--- Benchmarking {name} ---")
            print(f"Description: {arch['description']}")

            summary, detailed_results = benchmark_circuit_builder(
                arch["builder"],
                arch["params"],
                shots=shots,
                repeats=repeats,
                postselect_bit=None,
                csv_out=f"quantum_{name.lower()}_results.csv"
            )

            # Extract resource counts from first run
            resources = detailed_results[0] if detailed_results else {}

            self.architectures[name] = {
                "summary": summary,
                "resources": resources,
                "description": arch["description"],
                "execution_time_s": summary["mean_wall_time_s"],
                "throughput_samples_s": summary["mean_throughput"],
                "qubits": resources.get("n_qubits", 0),
                "single_qubit_gates": resources.get("one_q_gates", 0),
                "two_qubit_gates": resources.get("two_q_gates_est", 0),
                "total_gates": resources.get("one_q_gates", 0) + resources.get("two_q_gates_est", 0),
                "success_probability": summary["mean_postselect_prob"],
                "device_used": self.device_mode  # Track which device was used
            }

            self._print_architecture_results(name, self.architectures[name])

        return self.architectures

    def _print_architecture_results(self, name, results):
        """Print formatted results for a quantum architecture"""
        print(f"Architecture: {name}")
        print(f"  Description: {results['description']}")
        print(f"  Device: {results.get('device_used', 'unknown')}")
        print(f"  Qubits: {results['qubits']}")
        print(f"  Gates: {results['single_qubit_gates']} 1Q + {results['two_qubit_gates']} 2Q = {results['total_gates']} total")
        print(f"  Execution Time: {results['execution_time_s']:.3f}s")
        print(f"  Throughput: {results['throughput_samples_s']:.0f} samples/s")
        print(f"  Success Probability: {results['success_probability']:.4f}")

    def export_quantum_results_to_csv(self, filename="quantum_results.csv"):
        """Export quantum results to CSV for cross-environment comparison"""
        import csv
        import os

        fieldnames = [
            'architecture', 'description', 'execution_time_s', 'throughput_samples_s',
            'qubits', 'single_qubit_gates', 'two_qubit_gates', 'total_gates',
            'success_probability', 'environment', 'device_used', 'timestamp'
        ]

        # Check if file exists to write header
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
                    'qubits': results['qubits'],
                    'single_qubit_gates': results['single_qubit_gates'],
                    'two_qubit_gates': results['two_qubit_gates'],
                    'total_gates': results['total_gates'],
                    'success_probability': results['success_probability'],
                    'environment': 'braket_quantum',
                    'device_used': results.get('device_used', self.device_mode),
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                })

        print(f"Quantum results exported to {filename}")

    def generate_quantum_report(self):
        """Generate quantum-only performance report"""
        print("\n" + "="*60)
        print("QUANTUM ARCHITECTURE BENCHMARK REPORT")
        print("="*60)
        print(f"Device: {self.device_mode}")

        print(f"{'Architecture':<25} {'Qubits':<8} {'Total Gates':<12} {'Time (s)':<10} {'Throughput':<12} {'Success Prob':<12}")
        print("-" * 85)

        for name, results in self.architectures.items():
            print(f"{name:<25} {results['qubits']:<8} {results['total_gates']:<12} "
                  f"{results['execution_time_s']:<10.3f} {results['throughput_samples_s']:<12.0f} "
                  f"{results['success_probability']:<12.4f}")

        # Architecture comparison insights
        print("\n--- Quantum Architecture Insights ---")
        fastest = min(self.architectures.items(), key=lambda x: x[1]['execution_time_s'])
        most_efficient = min(self.architectures.items(), key=lambda x: x[1]['total_gates'])
        highest_throughput = max(self.architectures.items(), key=lambda x: x[1]['throughput_samples_s'])

        print(f"Fastest: {fastest[0]} ({fastest[1]['execution_time_s']:.3f}s)")
        print(f"Most Gate-Efficient: {most_efficient[0]} ({most_efficient[1]['total_gates']} gates)")
        print(f"Highest Throughput: {highest_throughput[0]} ({highest_throughput[1]['throughput_samples_s']:.0f} samples/s)")

# Usage examples for different devices
def run_benchmarks_on_different_devices():
    """Example of running benchmarks on different quantum devices"""

    devices_to_test = [
        ("local", "Local Simulator (free)"),
        ("sv1", "AWS State Vector Simulator"),
        ("tn1", "AWS Tensor Network Simulator"),
        ("ionq_aria", "IonQ Aria-1 QPU"),
        ("ionq_forte", "IonQ Forte QPU"),
        ("aqt_ibex", "AQT Ibex QPU")
    ]

    all_results = {}

    for device_mode, description in devices_to_test:
        print(f"\n{'='*60}")
        print(f"RUNNING ON: {description} ({device_mode})")
        print(f"{'='*60}")

        try:
            benchmark = QuantumArchitectureBenchmark(device_mode=device_mode)
            results = benchmark.benchmark_quantum_architectures(shots=100, repeats=1)  # Reduced for demo
            benchmark.generate_quantum_report()
            benchmark.export_quantum_results_to_csv(f"quantum_results_{device_mode}.csv")
            all_results[device_mode] = results
        except Exception as e:
            print(f"Failed to run on {device_mode}: {e}")

    return all_results

if __name__ == "__main__":
    # Option 1: Run on specific device
    device = "local"  # Change to "sv1", "ionq_aria", etc.
    quantum_benchmark = QuantumArchitectureBenchmark(device_mode=device)
    quantum_results = quantum_benchmark.benchmark_quantum_architectures(shots=1000, repeats=3)
    quantum_benchmark.generate_quantum_report()
    quantum_benchmark.export_quantum_results_to_csv("quantum_results.csv")
