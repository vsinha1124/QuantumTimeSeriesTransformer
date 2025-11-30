# Simple usage
from quantum_perf import run_quantum_perf_benchmark
from classical_perf import run_classical_perf_benchmark

# Run on different hardware
quantum_results = run_quantum_perf_benchmark(device_mode="local", shots=2000)
classical_results = run_classical_perf_benchmark(device="cuda")
