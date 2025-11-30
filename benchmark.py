# benchmarking_helper.py
# drop into same folder as your quixer script and import your build_* functions

import csv
import math
import time

import numpy as np
from braket.devices import LocalSimulator


# ---------- Helper: parse common Braket counts formats ----------
def parse_counts_and_detect_order(counts):
    # Supports string-key format {'010': n} or nested tuple format {(labels): {(bits): n}}
    if all(isinstance(k, str) and set(k) <= {"0", "1"} for k in counts.keys()):
        return counts, None
    # Nested form: keys are tuples of labels, values are dicts of bit tuples
    first_key = next(iter(counts))
    first_val = counts[first_key]
    if isinstance(first_key, tuple) and isinstance(first_val, dict):
        label_order = list(first_key)
        flat = {}
        for labels, inner in counts.items():
            for bits_tuple, c in inner.items():
                flat["".join(bits_tuple)] = flat.get("".join(bits_tuple), 0) + c
        return flat, label_order
    # fallback: coerce to strings
    return {str(k): v for k, v in counts.items()}, None


# ---------- Helper: compute Z expectation and variance for a single qubit ----------
def z_stats_from_flat_counts(flat_counts, bit_index):
    # Returns mean (⟨Z⟩), std, and sample_count (sum of counts)
    total = 0
    zsum = 0
    z2sum = 0
    for bstr, c in flat_counts.items():
        if bit_index >= len(bstr):
            continue
        bit = bstr[bit_index]
        val = 1 if bit == "0" else -1  # map to +1/-1
        total += c
        zsum += val * c
        z2sum += (val * val) * c  # equals c, but keep formulaic
    if total == 0:
        return 0.0, 0.0, 0
    mean = zsum / total
    # variance of discrete distribution (population variance)
    var = (z2sum / total) - (mean**2)
    std = math.sqrt(max(var, 0.0))
    return mean, std, total


# ---------- Helper: gate & qubit resource counts (best-effort) ----------
def resource_counts_from_circuit(circ):
    """
    Best-effort extraction. Braket Circuit objects expose `instructions`.
    Each instruction typically has .operator and .targets. This function
    attempts to count 1Q and 2Q gates and unique qubits used.
    """
    one_q = 0
    two_q = 0
    used_qubits = set()
    try:
        for instr in circ.instructions:
            targets = instr.target  # or .targets in some SDK versions
            # normalize to tuple
            if isinstance(targets, int):
                targets = (targets,)
            # record used qubits
            for t in targets:
                used_qubits.add(int(t))
            # count by arity
            if len(targets) == 1:
                one_q += 1
            elif len(targets) >= 2:
                # consider multi-target as two-qubit for cost
                two_q += 1
    except Exception:
        # fallback if instruct structure differs
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
    postselect_bit=None,  # set to integer index of control bit to postselect on (or None)
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
    sim = LocalSimulator()
    rows = []
    # run repeats to measure timing jitter & SEM
    for r in range(repeats):
        # build fresh circuit each repeat to capture any randomness in builder
        circ = circuit_builder_fn(**builder_kwargs)

        # resource counts (best-effort)
        resources = resource_counts_from_circuit(circ)

        # run & time
        t0 = time.time()
        task = sim.run(circ, shots=shots)
        result = task.result()
        t1 = time.time()
        wall_time = t1 - t0

        raw_counts = result.measurement_counts
        flat_counts, label_order = parse_counts_and_detect_order(raw_counts)

        # detect postselection index
        if label_order is not None and postselect_bit is None:
            # user passed None but we can auto-detect if 'lcu' or 'ctrl' exists
            if "lcu" in label_order:
                post_idx = label_order.index("lcu")
            elif "ctrl" in label_order:
                post_idx = label_order.index("ctrl")
            else:
                post_idx = None
        else:
            post_idx = postselect_bit

        # compute postselection stats
        if post_idx is not None:
            # total shots is sum flat_counts
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

        # compute Z stats on data qubits (assume data qubits at left-most indexes)
        # user can post-process differently if their bit-ordering differs
        z0_mean, z0_std, z0_N = z_stats_from_flat_counts(kept, 0)
        z1_mean, z1_std, z1_N = z_stats_from_flat_counts(kept, 1)

        # SEM (standard error of mean) for each qubit ⟨Z⟩: std/sqrt(N_eff)
        sem_z0 = (z0_std / math.sqrt(z0_N)) if z0_N > 0 else float("nan")
        sem_z1 = (z1_std / math.sqrt(z1_N)) if z1_N > 0 else float("nan")

        # effective throughput (useful samples / second)
        effective_samples = shots * p_succ
        throughput = effective_samples / wall_time if wall_time > 0 else float("inf")

        # estimate shots needed for target precision eps (binary var <= 1)
        def shots_needed_for_eps(eps, var_est, p_succ_local):
            # N_eff = var / eps^2, shots_needed = N_eff / p_succ
            if p_succ_local <= 0:
                return float("inf")
            N_eff = var_est / (eps**2)
            return math.ceil(N_eff / p_succ_local)

        shots_for_0_01 = shots_needed_for_eps(
            0.01, z0_std**2 if z0_N > 0 else 0.25, p_succ
        )

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
            f"[run {r}] p_succ={p_succ:.4f}, wall_time={wall_time:.3f}s, z0={z0_mean:.4f}±{sem_z0:.4f}"
        )

    # aggregate across repeats (mean ± std)
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

    # write detailed rows to CSV (append mode)
    fieldnames = list(rows[0].keys())
    with open(csv_out, "a", newline="") as f:
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
    # import your circuit builders here (example names)
    from quixer_benchmark import (
        build_quixer_mini_lcu,
        build_quixer_mini_with_qsvt_full_lcu,
        build_quixer_mini_with_qsvt_U,
    )

    # Minimal LCU benchmark
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
