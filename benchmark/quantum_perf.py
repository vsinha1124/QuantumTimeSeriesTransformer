# quantum_perf_benchmark.py
# Quantum benchmarking for all 3 circuit variants with JSON output

import time
import json
from datetime import datetime
from benchmark import get_device, resource_counts_from_circuit

from quixer_architecture import (
    build_quixer_mini_lcu,
    build_quixer_mini_with_qsvt_U,
    build_quixer_mini_with_qsvt_full_lcu
)


def get_quantum_transformer_time(
    circuit_builder, builder_kwargs, shots=1000, device_mode="local"
):
    """
    Benchmark execution time for a single quantum circuit variant.
    """
    try:
        # Build circuit
        circuit = circuit_builder(**builder_kwargs)

        # Select device
        device = get_device()

        start = time.time()
        task = device.run(circuit, shots=shots)
        result = task.result()
        end = time.time()

        resources = resource_counts_from_circuit(circuit)
        exec_time = end - start

        return {
            "execution_time_seconds": exec_time,
            "shots": shots,
            "effective_samples": shots,
            "throughput_samples_per_second": shots / exec_time,
            "circuit_qubits": resources["n_qubits"],
            "circuit_gates": resources["one_q_gates"] + resources["two_q_gates_est"],
            "device_mode": device_mode,
            "status": "success"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "execution_time_seconds": 0,
            "shots": shots,
            "effective_samples": 0,
            "throughput_samples_per_second": 0,
            "circuit_qubits": 0,
            "circuit_gates": 0,
            "device_mode": device_mode
        }


def run_quantum_benchmarks(shots=1000, device_mode="local"):
    """Run all quantum circuit variants and return JSON results"""


    # Define all circuit variants to test
    circuit_variants = [
        {
            "name": "LCU-only",
            "builder": build_quixer_mini_lcu,
            "kwargs": {
                "token0_angles": {"a0": 0.3, "a1": -0.2, "b0": 0.5, "b1": -0.1},
                "token1_angles": {"a0": -0.4, "a1": 0.6, "b0": -0.7, "b1": 0.2},
                "gamma": 0.7,
                "encode_angles": {"x0": 0.2, "x1": -0.1},
                "measure_all": True,
            }
        },
        {
            "name": "QSVT-U",
            "builder": build_quixer_mini_with_qsvt_U,
            "kwargs": {
                "token_angles": {"a0": 0.3, "a1": 0.5, "b0": -0.2, "b1": 0.4},
                "encode_angles": {"x0": 0.2, "x1": -0.1},
                "qsvt_phis": (0.3, 0.9, -0.4),
                "measure_all": True,
            }
        },
        {
            "name": "QSVT-Full",
            "builder": build_quixer_mini_with_qsvt_full_lcu,
            "kwargs": {
                "token0_angles": {"a0": 0.3, "a1": -0.2, "b0": 0.5, "b1": -0.1},
                "token1_angles": {"a0": -0.4, "a1": 0.6, "b0": -0.7, "b1": 0.2},
                "gamma": 0.7,
                "encode_angles": {"x0": 0.2, "x1": -0.1},
                "qsvt_phis": (0.3, 0.9, -0.4),
                "measure_all": True,
            }
        }
    ]

    print("\n🧪 QUANTUM TRANSFORMER BENCHMARKS")
    print("=================================")

    results = {
        "timestamp": datetime.now().isoformat(),
        "device_mode": device_mode,
        "shots": shots,
        "circuits": {}
    }

    for variant in circuit_variants:
        print(f"\n🔹 Testing {variant['name']}...")

        result = get_quantum_transformer_time(
            variant["builder"],
            variant["kwargs"],
            shots=shots,
            device_mode=device_mode
        )

        # Add variant info to result
        result["circuit_name"] = variant["name"]
        result["estimated_parameters"] = estimate_quixer_parameters(variant["builder"], variant["kwargs"])

        results["circuits"][variant["name"]] = result

        if result["status"] == "success":
            print(f"   ✅ Time: {result['execution_time_seconds']:.4f}s")
            print(f"   ✅ Throughput: {result['throughput_samples_per_second']:.0f} samples/s")
            print(f"   ✅ Qubits: {result['circuit_qubits']}, Gates: {result['circuit_gates']}")
        else:
            print(f"   ❌ Error: {result['error']}")

    return results


def estimate_quixer_parameters(circuit_builder, builder_kwargs):
    """Estimate trainable parameters for Quixer circuits"""
    param_count = 0

    # Count all angles in kwargs
    for key, value in builder_kwargs.items():
        if isinstance(value, dict) and any(k in key for k in ['angles', 'token']):
            param_count += len(value)
        elif key == 'gamma':
            param_count += 1
        elif key == 'qsvt_phis' and isinstance(value, (tuple, list)):
            param_count += len(value)
        elif key == 'encode_angles' and isinstance(value, dict):
            param_count += len(value)

    return param_count


def save_quantum_results(results, filename=None):
    """Save quantum benchmark results to JSON file"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"quantum_benchmark_results_{timestamp}.json"

    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Quantum results saved to: {filename}")
    return filename


if __name__ == "__main__":
    # Run all quantum benchmarks
    quantum_results = run_quantum_benchmarks(shots=1000, device_mode="local")

    # Save to JSON
    save_quantum_results(quantum_results)

    print(f"\n Quantum Benchmark Summary:")
    print(f"   Total circuits tested: {len(quantum_results['circuits'])}")
    print(f"   Successful: {sum(1 for r in quantum_results['circuits'].values() if r['status'] == 'success')}")
