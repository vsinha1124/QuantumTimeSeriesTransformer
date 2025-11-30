# quantum_perf_benchmark.py
# Simplified quantum benchmark using generic framework

from performance_benchmark import run_hardware_agnostic_benchmark, save_benchmark_results
from quantum_runner import QuantumBenchmarkRunner, get_quantum_model_configs

def run_quantum_perf_benchmark(device_mode="local", shots=1000):
    """Run quantum performance benchmarks"""
    print("🧪 QUANTUM PERFORMANCE BENCHMARK")
    print("================================")

    # Setup configurations
    runner = QuantumBenchmarkRunner()
    model_configs = get_quantum_model_configs()
    hardware_config = {"device_mode": device_mode}
    input_config = {"shots": shots}

    # Run benchmarks
    results = run_hardware_agnostic_benchmark(
        runner, model_configs, hardware_config, input_config
    )

    # Save results
    filename = save_benchmark_results(results, "quantum")

    print("\nQuantum benchmark completed!")
    print(f"   Circuits tested: {len(results['benchmarks'])}")
    print(f"   Results saved: {filename}")

    return results

if __name__ == "__main__":
    run_quantum_perf_benchmark(device_mode="local", shots=1000)
