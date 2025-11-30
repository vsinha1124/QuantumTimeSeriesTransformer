# classical_perf_benchmark.py
# Classical benchmarking for all model variants with JSON output

import time
import json
import torch
from datetime import datetime
from classical_benchmark import SimpleAttentionHead, MiniTransformerLayer


def get_classical_transformer_time(model, input_tensor, model_name=""):
    """
    Benchmark a classical transformer component on CPU or GPU.
    """
    try:
        model.eval()

        if torch.cuda.is_available():
            model = model.cuda()
            input_tensor = input_tensor.cuda()

        # Warm-up
        with torch.no_grad():
            _ = model(input_tensor)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Timed execution
        start = time.time()
        with torch.no_grad():
            output = model(input_tensor)
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
            "hidden_dimension": d_model,
            "model_parameters": sum(p.numel() for p in model.parameters()),
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "status": "success",
            "model_name": model_name
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "execution_time_seconds": 0,
            "tokens_processed": 0,
            "throughput_tokens_per_second": 0,
            "model_parameters": 0,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "model_name": model_name
        }


def run_classical_benchmarks():
    """Run all classical model variants and return JSON results"""

    # Test configurations for fair comparison
    test_configs = [
        {
            "name": "SmallAttentionHead",
            "model_class": SimpleAttentionHead,
            "args": {"d_model": 16},  # ~768 params
            "input_shape": (1, 2, 16)  # batch, seq_len, d_model
        },
        {
            "name": "MediumAttentionHead",
            "model_class": SimpleAttentionHead,
            "args": {"d_model": 32},  # ~3K params
            "input_shape": (1, 2, 32)
        },
        {
            "name": "SmallTransformer",
            "model_class": MiniTransformerLayer,
            "args": {"d_model": 32, "num_heads": 4},  # ~15K params
            "input_shape": (1, 2, 32)
        },
        {
            "name": "MediumTransformer",
            "model_class": MiniTransformerLayer,
            "args": {"d_model": 64, "num_heads": 8},  # ~60K params
            "input_shape": (1, 2, 64)
        }
    ]

    print("\n🖥️  CLASSICAL TRANSFORMER BENCHMARKS")
    print("===================================")

    results = {
        "timestamp": datetime.now().isoformat(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "models": {}
    }

    for config in test_configs:
        print(f"\n🔹 Testing {config['name']}...")

        try:
            # Create model and input
            model = config["model_class"](**config["args"])
            input_tensor = torch.randn(*config["input_shape"])

            result = get_classical_transformer_time(model, input_tensor, config["name"])
            results["models"][config["name"]] = result

            if result["status"] == "success":
                print(f"   ✅ Time: {result['execution_time_seconds']:.6f}s")
                print(f"   ✅ Throughput: {result['throughput_tokens_per_second']:.0f} tokens/s")
                print(f"   ✅ Params: {result['model_parameters']:,}")
                print(f"   ✅ Device: {result['device']}")
            else:
                print(f"   ❌ Error: {result['error']}")

        except Exception as e:
            error_result = {
                "status": "error",
                "error": str(e),
                "model_name": config["name"]
            }
            results["models"][config["name"]] = error_result
            print(f"   ❌ Setup Error: {e}")

    return results


def save_classical_results(results, filename=None):
    """Save classical benchmark results to JSON file"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"classical_benchmark_results_{timestamp}.json"

    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Classical results saved to: {filename}")
    return filename


if __name__ == "__main__":
    # Run all classical benchmarks
    classical_results = run_classical_benchmarks()

    # Save to JSON
    save_classical_results(classical_results)

    print(f"\n📊 Classical Benchmark Summary:")
    print(f"   Total models tested: {len(classical_results['models'])}")
    print(f"   Successful: {sum(1 for r in classical_results['models'].values() if r['status'] == 'success')}")
