# benchmark.py
# Unified benchmarking helper for Quixer-Mini circuits.
# Choose LocalSimulator, SV1, TN1, or real QPUs (IonQ / AQT) via DEVICE_MODE.

import csv
import math
import time

import numpy as np
from braket.devices import LocalSimulator
from braket.aws import AwsDevice, AwsSession

#################################################################################
###########################  DEVICE  SELECTION  CODE  ###########################
#################################################################################

# ---------- Device selection config ----------
# Change this ONE variable to switch where circuits run.
# Options:
#   "local"         - Braket LocalSimulator (fast, free)
#   "sv1"           - AWS state-vector simulator
#   "tn1"           - AWS tensor-network simulator
#   "ionq_aria"     - IonQ Aria-1 trapped-ion QPU (us-east-1)
#   "ionq_forte"    - IonQ Forte trapped-ion QPU (us-east-1)
#   "aqt_ibex"      - AQT Ibex-Q1 trapped-ion QPU (eu-north-1)
DEVICE_MODE = "local"


def get_device():
    """
    Returns a Braket device object based on DEVICE_MODE,
    with automatic region routing.
    """

    # Local simulator: no region needed
    if DEVICE_MODE == "local":
        return LocalSimulator()

    # AWS-managed simulators (pick a convenient region)
    if DEVICE_MODE == "sv1":
        session = AwsSession(region="us-east-1")
        return AwsDevice(
            "arn:aws:braket:::device/quantum-simulator/amazon/sv1",
            aws_session=session,
        )

    if DEVICE_MODE == "tn1":
        session = AwsSession(region="us-east-1")
        return AwsDevice(
            "arn:aws:braket:::device/quantum-simulator/amazon/tn1",
            aws_session=session,
        )

    # IonQ hardware (us-east-1)
    if DEVICE_MODE == "ionq_aria":
        session = AwsSession(region="us-east-1")
        return AwsDevice(
            "arn:aws:braket:us-east-1::device/qpu/ionq/Aria-1",
            aws_session=session,
        )

    if DEVICE_MODE == "ionq_forte":
        session = AwsSession(region="us-east-1")
        return AwsDevice(
            "arn:aws:braket:us-east-1::device/qpu/ionq/Forte",
            aws_session=session,
        )

    # AQT Ibex-Q1 (eu-north-1)
    if DEVICE_MODE == "aqt_ibex":
        session = AwsSession(region="eu-north-1")
        return AwsDevice(
            "arn:aws:braket:eu-north-1::device/qpu/aqt/Ibex-Q1",
            aws_session=session,
        )

    raise ValueError(f"Unknown DEVICE_MODE: {DEVICE_MODE}")


def assert_device_ready(device):
    """
    Raise a helpful error if the device is offline or not usable right now.
    """
    try:
        status = device.status
    except Exception as e:
        raise RuntimeError(f"Could not query device status: {e}")

    if str(status).upper() not in ("ONLINE", "AVAILABLE"):
        raise RuntimeError(f"Device {getattr(device, 'arn', device)} is not available. Status: {status}")


def assert_circuit_fits_device(circ, device):
    """
    Ensure the circuit's qubit count does not exceed the device capacity.
    If properties are unavailable, this silently passes.
    """
    # count distinct qubits in the circuit
    qubits = set()
    try:
        for instr in circ.instructions:
            targets = getattr(instr, "target", getattr(instr, "targets", None))
            if isinstance(targets, int):
                targets = (targets,)
            elif targets is None:
                targets = ()
            for t in targets:
                try:
                    qubits.add(int(t))
                except Exception:
                    pass
    except Exception:
        # if instructions isn't present, skip check
        return

    n_circ = len(qubits)
    # try to read device qubit count
    try:
        n_dev = device.properties.paradigm.qubitCount
    except Exception:
        return

    if n_circ > n_dev:
        raise RuntimeError(
            f"Circuit uses {n_circ} qubits but device {getattr(device, 'arn', device)} only has {n_dev}."
        )

##################################################################################
#######################  END  OF  DEVICE  SELECTION  CODE  #######################
##################################################################################

# ---------- Helper: parse common Braket counts formats ----------
def parse_counts_and_detect_order(counts):
    """
    Normalize Braket measurement_counts into a flat dict plus optional label order.

    Supports:
      - string-key format: {'010': n}
      - nested tuple format: {(labels): {(bits): n}}
    """
    # Already flat string->int with only 0/1 keys?
    try:
        if all(isinstance(k, str) and set(k) <= {"0", "1"} for k in counts.keys()):
            return counts, None
    except Exception:
        pass

    # Nested dict keyed by tuples of labels, values are dicts of bit tuples
    first_key = next(iter(counts))
    first_val = counts[first_key]
    if isinstance(first_key, tuple) and isinstance(first_val, dict):
        label_order = list(first_key)
        flat = {}
        for labels, inner in counts.items():
            for bits_tuple, c in inner.items():
                bitstring = "".join(bits_tuple)
                flat[bitstring] = flat.get(bitstring, 0) + c
        return flat, label_order

    # Fallback: coerce everything to strings
    flat = {}
    for k, v in counts.items():
        flat[str(k)] = v
    return flat, None


# ---------- Helper: compute Z expectation and variance for a single qubit ----------
def z_stats_from_flat_counts(flat_counts, bit_index):
    """
    Compute Z statistics for a given bit index in flat_counts.

    Returns (mean_Z, std_Z, sample_count).
    """
    total = 0
    zsum = 0.0
    z2sum = 0.0
    for bstr, c in flat_counts.items():
        if bit_index >= len(bstr):
            continue
        bit = bstr[bit_index]
        val = 1.0 if bit == "0" else -1.0
        total += c
        zsum += val * c
        z2sum += (val * val) * c  # equals c, but keep formulaic
    if total == 0:
        return 0.0, 0.0, 0
    mean = zsum / total
    var = (z2sum / total) - (mean**2)
    std = math.sqrt(max(var, 0.0))
    return mean, std, total


# ---------- Helper: gate & qubit resource counts (best-effort) ----------
def resource_counts_from_circuit(circ):
    """
    Best-effort extraction of resource counts.

    Braket Circuit objects expose `instructions`.
    Each instruction typically has .operator and .target(s).
    This function attempts to count 1Q and 2Q gates and unique qubits used.
    """
    one_q = 0
    two_q = 0
    used_qubits = set()
    try:
        for instr in circ.instructions:
            targets = getattr(instr, "target", getattr(instr, "targets", None))
            if isinstance(targets, int):
                targets = (targets,)
            elif targets is None:
                targets = ()
            for t in targets:
                used_qubits.add(int(t))
            if len(targets) == 1:
                one_q += 1
            elif len(targets) >= 2:
                two_q += 1
    except Exception:
        pass
    return {
        "n_qubits": len(used_qubits),
        "one_q_gates": one_q,
        "two_q_gates_est": two_q,
    }


# ---------- Core benchmark function ----------
def benchmark_circuit_builder(
    circuit_builder_fn,
    builder_kwargs,
    shots=4000,
    repeats=3,
    postselect_bit=None,  # index of control bit to postselect on (or None)
    postselect_value="0",
    csv_out="benchmark_results.csv",
):
    """
    circuit_builder_fn: callable that returns a braket.Circuit when called with **builder_kwargs
    builder_kwargs: dict passed to circuit_builder_fn
    shots: shots per run
    repeats: how many independent runs (for timing + SEM aggregation)
    postselect_bit: index in bitstring to postselect on, or None (no postselection)
    """

    device = get_device()
    # For LocalSimulator, status may not exist; guard in try/except
    try:
        assert_device_ready(device)
    except Exception as e:
        # It's fine to continue for LocalSimulator if status is absent
        if not isinstance(device, LocalSimulator):
            raise

    rows = []
    for r in range(repeats):
        # Build fresh circuit each repeat
        circ = circuit_builder_fn(**builder_kwargs)

        # Sanity-check qubit count vs device capacity (if available)
        assert_circuit_fits_device(circ, device)

        # Resource counts (best-effort)
        resources = resource_counts_from_circuit(circ)

        # Run & time
        t0 = time.time()
        task = device.run(circ, shots=shots)
        result = task.result()
        t1 = time.time()
        wall_time = t1 - t0

        raw_counts = result.measurement_counts
        flat_counts, label_order = parse_counts_and_detect_order(raw_counts)

        # Detect postselection index
        if label_order is not None and postselect_bit is None:
            # auto-detect if 'lcu' or 'ctrl' exists
            if "lcu" in label_order:
                post_idx = label_order.index("lcu")
            elif "ctrl" in label_order:
                post_idx = label_order.index("ctrl")
            else:
                post_idx = None
        else:
            post_idx = postselect_bit

        # Compute postselection stats
        if post_idx is not None:
            total_shots = sum(flat_counts.values())
            kept = {
                b: c
                for b, c in flat_counts.items()
                if len(b) > post_idx and b[post_idx] == postselect_value
            }
            kept_shots = sum(kept.values())
            p_succ = kept_shots / total_shots if total_shots > 0 else 0.0
        else:
            kept = flat_counts
            total_shots = sum(flat_counts.values())
            kept_shots = total_shots
            p_succ = 1.0

        # Compute Z stats on data qubits (assume data qubits at left-most indexes)
        z0_mean, z0_std, z0_N = z_stats_from_flat_counts(kept, 0)
        z1_mean, z1_std, z1_N = z_stats_from_flat_counts(kept, 1)

        sem_z0 = (z0_std / math.sqrt(z0_N)) if z0_N > 0 else float("nan")
        sem_z1 = (z1_std / math.sqrt(z1_N)) if z1_N > 0 else float("nan")

        # Effective throughput (useful samples / second)
        effective_samples = shots * p_succ
        throughput = effective_samples / wall_time if wall_time > 0 else float("inf")

        # Estimate shots needed for target precision eps (for z0)
        def shots_needed_for_eps(eps, var_est, p_succ_local):
            if p_succ_local <= 0:
                return float("inf")
            N_eff = var_est / (eps**2)
            return math.ceil(N_eff / p_succ_local)

        var0_est = z0_std**2 if z0_N > 0 else 0.25
        shots_for_0_01 = shots_needed_for_eps(0.01, var0_est, p_succ)

        row = {
            "repeat": r,
            "shots_requested": shots,
            "shots_total_recorded": total_shots,
            "shots_kept": kept_shots,
            "postselect_prob": p_succ,
            "wall_time_s": wall_time,
            "effective_useful_samples": effective_samples,
            "throughput_useful_s_per_s": throughput,
            "z0_mean": z0_mean,
            "z0_sem": sem_z0,
            "z1_mean": z1_mean,
            "z1_sem": sem_z1,
            "z0_N": z0_N,
            "z1_N": z1_N,
            "shots_needed_for_0.01_on_z0": shots_for_0_01,
        }
        row.update(resources)
        rows.append(row)

        print(
            f"[run {r}] p_succ={p_succ:.4f}, wall_time={wall_time:.3f}s, "
            f"z0={z0_mean:.4f}±{sem_z0:.4f}"
        )

    # Aggregate across repeats (mean ± std)
    def agg(key):
        vals = [row[key] for row in rows]
        return np.mean(vals), np.std(vals)

    summary = {
        "variant": circuit_builder_fn.__name__,
        "mean_postselect_prob": agg("postselect_prob")[0],
        "std_postselect_prob": agg("postselect_prob")[1],
        "mean_wall_time_s": agg("wall_time_s")[0],
        "std_wall_time_s": agg("wall_time_s")[1],
        "mean_throughput": agg("throughput_useful_s_per_s")[0],
        "std_throughput": agg("throughput_useful_s_per_s")[1],
    }

    # Write detailed rows to CSV (append mode)
    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_out, "a+", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            # write header only if empty file
            f.seek(0)
            if f.read(1) == "":
                f.seek(0)
                writer.writeheader()
            for rrow in rows:
                writer.writerow(rrow)

    return summary, rows


# ---------- Example usage ----------
if __name__ == "__main__":
    from quixer_benchmark import (
        build_quixer_mini_lcu,
        build_quixer_mini_with_qsvt_full_lcu,
        build_quixer_mini_with_qsvt_U,
    )

    builder = build_quixer_mini_lcu
    kwargs = {
        "token0_angles": {"a0": 0.3, "a1": -0.2, "b0": 0.5, "b1": -0.1},
        "token1_angles": {"a0": -0.4, "a1": 0.6, "b0": -0.7, "b1": 0.2},
        "gamma": 0.7,
        "encode_angles": {"x0": 0.2, "x1": -0.1},
        "measure_all": True,
    }
    summary, rows = benchmark_circuit_builder(
        builder,
        kwargs,
        shots=2000,
        repeats=3,
        postselect_bit=-1,
        csv_out="bench_lcu.csv",
    )
    print("Summary:", summary)
