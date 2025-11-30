# classical_runner.py
# Classical-specific benchmark runner

from performance_benchmark import BenchmarkRunner, BenchmarkResult
from classical_architecture import SimpleAttentionHead, MiniTransformerLayer
import torch
import time


class ClassicalBenchmarkRunner(BenchmarkRunner):
    """Classical model benchmark runner"""

    def __init__(self):
        self.device = "auto"

    def setup(self, hardware_config):
        """Setup classical hardware"""
        self.device = hardware_config.get("device", "auto")
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"   Classical device: {self.device}")

    def run_benchmark(self, model_config, input_config):
        """Run classical model benchmark"""
        model = model_config["model"]
        input_tensor = input_config["input_data"]

        # Move to device
        model.to(self.device)
        input_tensor = input_tensor.to(self.device)
        model.eval()

        # Warm-up
        with torch.no_grad():
            _ = model(input_tensor)

        if self.device == "cuda":
            torch.cuda.synchronize()

        # Timed execution
        start_time = time.time()
        with torch.no_grad():
            output = model(input_tensor)
        if self.device == "cuda":
            torch.cuda.synchronize()
        end_time = time.time()

        execution_time = end_time - start_time

        # Calculate throughput (tokens/second)
        batch_size, seq_len, d_model = input_tensor.shape
        tokens_processed = batch_size * seq_len
        throughput = tokens_processed / execution_time

        parameters = sum(p.numel() for p in model.parameters())

        metadata = {
            "batch_size": batch_size,
            "sequence_length": seq_len,
            "hidden_dimension": d_model,
            "device": self.device
        }

        return BenchmarkResult(
            architecture_type="classical",
            model_name=model_config["name"],
            execution_time=execution_time,
            throughput=throughput,
            parameters=parameters,
            metadata=metadata
        )

    def teardown(self):
        """Classical cleanup"""
        if self.device == "cuda":
            torch.cuda.empty_cache()


def get_classical_model_configs():
    """Define all classical model variants to test"""
    # Parameter-matched equivalents to quantum circuits
    return [
        {
            "name": "TinyAttention",
            "model": SimpleAttentionHead(d_model=2),  # ~12 params (matches LCU)
            "description": "Matches LCU-only parameter count"
        },
        {
            "name": "SmallAttention",
            "model": SimpleAttentionHead(d_model=3),  # ~27 params (matches QSVT-U)
            "description": "Matches QSVT-U parameter count"
        },
        {
            "name": "MiniTransformer",
            "model": MiniTransformerLayer(d_model=4, num_heads=2),  # ~100 params
            "description": "Matches QSVT-Full capability"
        }
    ]
