# Quixer-Mini: 4-qubit LCU-based "attention" block on AWS Braket
# --------------------------------------------------------------
# Qubit layout:
#   q0, q1 : data register (2 qubits)
#   q2     : control (LCU mixing weights)

from braket.circuits import Circuit
from braket.aws import AwsDevice
from braket.devices import LocalSimulator
import numpy as np


# -----------------------------
# 1. Token PQCs (unitary blocks)
# -----------------------------
def add_controlled_token_pqc(circ, data_qubits, control_qubit, angles):
    """
    Add a VERY small token-specific PQC on `data_qubits`,
    controlled by `control_qubit`.

    angles: dict with keys "a0", "a1", "b0", "b1"
        - a* : first layer RY angles
        - b* : second layer RY angles
    Structure (uncontrolled version) would be:
        RY(a0) on q0
        RY(a1) on q1
        CNOT(q0 -> q1)
        RY(b0) on q0
        RY(b1) on q1
    Here we make the RY gates controlled by control_qubit.
    The CNOT between data qubits is unconditional (same for both tokens).
    """
    q0, q1 = data_qubits

    # Layer 1: controlled RY on data qubits
    circ.ry(q0, angles["a0"], control=control_qubit)
    circ.ry(q1, angles["a1"], control=control_qubit)

    # Entangling layer on data (uncontrolled)
    circ.cnot(q0, q1)

    # Layer 2: controlled RY on data qubits
    circ.ry(q0, angles["b0"], control=control_qubit)
    circ.ry(q1, angles["b1"], control=control_qubit)

    return circ


# -------------------------------------------
# 2. Build one Quixer-Mini LCU-only circuit
# -------------------------------------------
def build_quixer_mini_circuit(
    token0_angles,
    token1_angles,
    gamma,
    encode_angles=None,
    measure_all=True,
):
    """
    Build a 4-qubit "Quixer-Mini" circuit implementing:
      - LCU mixing of two token unitaries U0, U1
      - Data register of 2 qubits
      - 1 control qubit

    token0_angles, token1_angles: dicts for add_controlled_token_pqc
    gamma: mixing angle; cos(gamma)=b0, sin(gamma)=b1
    encode_angles: optional dict {"x0":..., "x1":...} to encode input on data
    """
    circ = Circuit()

    # Qubit indices
    q0, q1 = 0, 1   # data
    qc        = 2   # control

    # (Optional) encode some classical input on data qubits
    if encode_angles is not None:
        circ.ry(q0, encode_angles.get("x0", 0.0))
        circ.ry(q1, encode_angles.get("x1", 0.0))

    # --- LCU preparation: create b0|0> + b1|1> on control ---
    # |0> --RY(2gamma)--> cos(gamma)|0> + sin(gamma)|1>
    circ.ry(qc, 2.0 * gamma)

    # --- Controlled U0 on data when control = |0> ---
    # Implement control-on-0 using X sandwich:
    #   X qc
    #   [control-on-1 version]
    #   X qc
    circ.x(qc)
    add_controlled_token_pqc(
        circ,
        data_qubits=(q0, q1),
        control_qubit=qc,
        angles=token0_angles,
    )
    circ.x(qc)

    # --- Controlled U1 on data when control = |1> ---
    add_controlled_token_pqc(
        circ,
        data_qubits=(q0, q1),
        control_qubit=qc,
        angles=token1_angles,
    )

    # --- Uncompute LCU superposition (optional but nice) ---
    circ.ry(qc, -2.0 * gamma)

    # --- Measurements ---
    if measure_all:
        circ.measure(q0, "d0")
        circ.measure(q1, "d1")
        circ.measure(qc, "ctrl")

    return circ


# --------------------------------
# 3. Example parameters + circuit
# --------------------------------

# Example token unitaries (just random-ish angles)
token0_angles = {
    "a0": 0.3,
    "a1": -0.2,
    "b0": 0.5,
    "b1": -0.1,
}

token1_angles = {
    "a0": -0.4,
    "a1": 0.6,
    "b0": -0.7,
    "b1": 0.2,
}

# Mixing weights: choose gamma so that
# b0 = cos(gamma), b1 = sin(gamma)
gamma = 0.7  # arbitrary; you can make this trainable later

# Example input encoding on data qubits
encode_angles = {"x0": 0.2, "x1": -0.1}

circuit = build_quixer_mini_circuit(
    token0_angles,
    token1_angles,
    gamma,
    encode_angles=encode_angles,
)

print("Quixer-Mini circuit:")
print(circuit)
print("Depth:", circuit.depth)


# -------------------------------
# 4. Run on a simulator (SV1 or local)
# -------------------------------
def run_on_sv1(circuit, shots=4000):
    device = AwsDevice("arn:aws:braket:::device/quantum-simulator/amazon/sv1")
    task = device.run(circuit, shots=shots)
    result = task.result()
    return result.measurement_counts


def run_on_local(circuit, shots=4000):
    sim = LocalSimulator()
    task = sim.run(circuit, shots=shots)
    result = task.result()
    return result.measurement_counts


counts = run_on_local(circuit, shots=4000)
print("Raw counts:", counts)


# --------------------------------------------
# 5. Postselect on successful LCU (ctrl = 0)
#    and compute <Z> on data qubits
# --------------------------------------------
def z_expectation_from_counts(counts, qubit_key):
    """
    Compute <Z> = P(0) - P(1) for a given measured bit (0 or 1).
    counts: dict mapping bitstring labels -> counts, e.g. {'000': 100, '010': 200, ...}
    qubit_key: index in the string; here keys are classical reg names,
               but we used 'd0', 'd1', 'ctrl', so we instead map by those keys.
    """
    # In this measurement layout, we have separate classical keys,
    # so the result.measurement_counts() returns dicts like:
    # {('d0','d1','ctrl'): count}. In practice with Braket, you'll
    # get something like {'000': 10, '010': 20, ...} if using a single measure call.
    # For simplicity in this script, we assume the keys are "d0d1ctrl"
    # with d0 as first bit, d1 second, ctrl third.
    # Adjust this function if the format differs in your environment.
    total = 0
    z = 0
    idx = qubit_key  # position in the bitstring: 0, 1, or 2
    for bitstring, c in counts.items():
        # bitstring is like '001' (as a string)
        bit = bitstring[idx]
        total += c
        if bit == "0":
            z += c
        else:
            z -= c
    return z / total if total > 0 else 0.0


# For Braket's default bit-order, d0,d1,ctrl will map to something like:
#   bitstring[0] -> d0, bitstring[1] -> d1, bitstring[2] -> ctrl
# Check `counts` keys once and adjust if needed.

# Postselect on ctrl = '0'
post_counts = {}
for bitstring, c in counts.items():
    # assuming bitstring is like 'd0d1ctrl' with ctrl as last bit
    if bitstring[-1] == "0":
        post_counts[bitstring] = c

print("Postselected counts (ctrl=0):", post_counts)

z0 = z_expectation_from_counts(post_counts, qubit_key=0)
z1 = z_expectation_from_counts(post_counts, qubit_key=1)
print("<Z0> =", z0)
print("<Z1> =", z1)
