# classical_perf_benchmark.py
# Simplified classical benchmark using generic framework

import torch
from performance_benchmark import run_hardware_agnostic_benchmark, save_benchmark_results
from classical_runner import ClassicalBenchmarkRunner, get_classical_model_configs

def run_classical_perf_benchmark(device="auto"):
    """Run classical performance benchmarks"""
    print("🖥️  CLASSICAL PERFORMANCE BENCHMARK")
    print("==================================")

    # Setup configurations
    runner = ClassicalBenchmarkRunner()
    model_configs = get_classical_model_configs()
    hardware_config = {"device": device}

    # Create input data (batch_size=1, seq_len=2, d_model=matches model)
    input_configs = []
    for model_config in model_configs:
        d_model = model_config["model"].d_model if hasattr(model_config["model"], 'd_model') else 4
        input_tensor = torch.randn(1, 2, d_model)  # batch, seq, dim
        input_configs.append({
            "input_data": input_tensor,
            "description": f"Random input for {model_config['name']}"
        })

    # Run benchmarks (each model with its own input)
    all_results = []
    for model_config, input_config in zip(model_configs, input_configs):
        results = run_hardware_agnostic_benchmark(
            runner, [model_config], hardware_config, input_config
        )
        all_results.extend(results["benchmarks"])

    # Combine results
    combined_results = {
        "timestamp": datetime.now().isoformat(),
        "hardware_config": hardware_config,
        "benchmarks": all_results
    }

    # Save results
    filename = save_benchmark_results(combined_results, "classical")

    print(f"\n📊 Classical benchmark completed!")
    print(f"   Models tested: {len(combined_results['benchmarks'])}")
    print(f"   Results saved: {filename}")

    return combined_results

if __name__ == "__main__":
    run_classical_perf_benchmark(device="auto")
