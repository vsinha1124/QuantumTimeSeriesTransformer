# quantum_runner.py
# Quantum-specific benchmark runner

from performance_benchmark import BenchmarkRunner, BenchmarkResult
from benchmark import get_device, resource_counts_from_circuit
import torch
import time


class QuantumBenchmarkRunner(BenchmarkRunner):
    """Quantum circuit benchmark runner"""

    def __init__(self):
        self.device = None
        self.device_mode = "local"

    def setup(self, hardware_config):
        """Setup quantum hardware"""
        self.device_mode = hardware_config.get("device_mode", "local")
        # Note: get_device() uses global DEVICE_MODE, so we might need to set it
        # Alternatively, we could modify benchmark.py to accept device_mode as parameter
        global DEVICE_MODE
        DEVICE_MODE = self.device_mode
        self.device = get_device()
        print(f"   Quantum device: {self.device_mode}")

    def run_benchmark(self, model_config, input_config):
        """Run quantum circuit benchmark"""
        circuit_builder = model_config["builder"]
        builder_kwargs = model_config["kwargs"]
        shots = input_config.get("shots", 1000)

        # Build and run circuit
        circuit = circuit_builder(**builder_kwargs)

        start_time = time.time()
        task = self.device.run(circuit, shots=shots)
        result = task.result()
        end_time = time.time()

        execution_time = end_time - start_time
        resources = resource_counts_from_circuit(circuit)

        # Calculate throughput (samples/second)
        throughput = shots / execution_time

        # Estimate parameters
        parameters = self._estimate_quixer_parameters(builder_kwargs)

        metadata = {
            "shots": shots,
            "qubits": resources["n_qubits"],
            "gates": resources["one_q_gates"] + resources["two_q_gates_est"],
            "device_mode": self.device_mode
        }

        return BenchmarkResult(
            architecture_type="quantum",
            model_name=model_config["name"],
            execution_time=execution_time,
            throughput=throughput,
            parameters=parameters,
            metadata=metadata
        )

    def _estimate_quixer_parameters(self, builder_kwargs):
        """Estimate trainable parameters for Quixer circuits"""
        param_count = 0
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

    def teardown(self):
        """Quantum cleanup (if needed)"""
        pass


def get_quantum_model_configs():
    """Define all quantum circuit variants to test"""
    from quixer_architecture import (
        build_quixer_mini_lcu,
        build_quixer_mini_with_qsvt_U,
        build_quixer_mini_with_qsvt_full_lcu
    )

    return [
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
