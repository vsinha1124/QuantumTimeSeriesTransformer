# Quixer-Mini with and without a 1-step QSVT-inspired block
# =========================================================
# This script gives you:
#   1) LCU-only Quixer-Mini circuit  (purely linear mixing)
#   2) QSVT-inspired Quixer-Mini circuit (adds ancilla nonlinearity)
#
# Runs entirely on Braket LocalSimulator; no AWS account needed.

from braket.circuits import Circuit
from braket.devices import LocalSimulator
import numpy as np


# ---------------------------------------------------------
# 1. Token PQCs (small learnable unitaries on the data regs)
# ---------------------------------------------------------
def add_token_pqc(circ, q0, q1, angles, control=None):
    """
    Very small two-qubit PQC:
        RY(a0) on q0
        RY(a1) on q1
        CNOT(q0 -> q1)
        RY(b0) on q0
        RY(b1) on q1

    If control is not None, RY gates are controlled by `control`.
    (This uses Braket's 'control' modifier, which is supported on LocalSimulator.)
    """
    a0, a1 = angles["a0"], angles["a1"]
    b0, b1 = angles["b0"], angles["b1"]

    if control is None:
        circ.ry(q0, a0)
        circ.ry(q1, a1)
    else:
        circ.ry(q0, a0, control=control)
        circ.ry(q1, a1, control=control)

    # Simple entangling layer (uncontrolled)
    circ.cnot(q0, q1)

    if control is None:
        circ.ry(q0, b0)
        circ.ry(q1, b1)
    else:
        circ.ry(q0, b0, control=control)
        circ.ry(q1, b1, control=control)

    return circ


def add_token_pqc_dagger(circ, q0, q1, angles, control=None):
    """
    Adjoint of add_token_pqc, used in the QSVT-inspired block.
    Reverse order of gates, negate angles.
    """
    a0, a1 = angles["a0"], angles["a1"]
    b0, b1 = angles["b0"], angles["b1"]

    # Reverse of second layer RYs
    if control is None:
        circ.ry(q1, -b1)
        circ.ry(q0, -b0)
    else:
        circ.ry(q1, -b1, control=control)
        circ.ry(q0, -b0, control=control)

    # Reverse of CNOT (self-inverse)
    circ.cnot(q0, q1)

    # Reverse of first layer RYs
    if control is None:
        circ.ry(q1, -a1)
        circ.ry(q0, -a0)
    else:
        circ.ry(q1, -a1, control=control)
        circ.ry(q0, -a0, control=control)

    return circ


# ----------------------------------------------
# 2. LCU-only Quixer-Mini (what you had before)
# ----------------------------------------------
def build_quixer_mini_lcu(
    token0_angles,
    token1_angles,
    gamma,
    encode_angles=None,
    measure_all=True,
):
    """
    4-qubit circuit:

        q0, q1 : data
        q2     : control for LCU

    Implements:
        A = b0 U0 + b1 U1   (linear combination of token unitaries)

    This is the *linear* part only (no QSVT yet).
    """
    circ = Circuit()
    q0, q1 = 0, 1
    qc = 2   # LCU control

    # (Optional) encode classical input
    if encode_angles is not None:
        circ.ry(q0, encode_angles.get("x0", 0.0))
        circ.ry(q1, encode_angles.get("x1", 0.0))

    # Prepare b0|0> + b1|1> on qc
    circ.ry(qc, 2.0 * gamma)

    # Controlled U0 when qc = 0  (X sandwich trick)
    circ.x(qc)
    add_token_pqc(circ, q0, q1, token0_angles, control=qc)
    circ.x(qc)

    # Controlled U1 when qc = 1
    add_token_pqc(circ, q0, q1, token1_angles, control=qc)

    # Uncompute
    circ.ry(qc, -2.0 * gamma)

    if measure_all:
        circ.measure(q0, "d0")
        circ.measure(q1, "d1")
        circ.measure(qc, "ctrl")

    return circ


# --------------------------------------------------
# 3. QSVT-inspired Quixer-Mini (adds ancilla block)
# --------------------------------------------------
def build_quixer_mini_with_qsvt(
    token_angles,
    encode_angles=None,
    qsvt_phis=(0.3, 0.7, -0.4),
    measure_all=True,
):
    """
    4-qubit circuit:

        q0, q1 : data
        q2     : ancilla (QSVT-style)
        q3     : (unused here, left for extension)

    We use ONE token unitary U (on q0,q1) as our "A".
    Then:

        - Encode input on data
        - Apply U once
        - Apply 1-step QSVT-inspired block:

            Rz(phi0) on ancilla
            controlled-U on data (control = ancilla)
            Ry(phi1) on ancilla
            controlled-U† on data (control = ancilla)
            Rz(phi2) on ancilla

      This implements a small polynomial in U: roughly p(U) = a U + b U^2.
    """
    circ = Circuit()
    q0, q1 = 0, 1     # data
    qa = 2            # ancilla for QSVT-style block

    phi0, phi1, phi2 = qsvt_phis

    # (Optional) encode input
    if encode_angles is not None:
        circ.ry(q0, encode_angles.get("x0", 0.0))
        circ.ry(q1, encode_angles.get("x1", 0.0))

    # 1) Apply U once to data (like a linear attention-ish transform)
    add_token_pqc(circ, q0, q1, token_angles, control=None)

    # 2) QSVT-inspired block with ancilla
    # Ancilla starts in |0>
    # Rz(phi0)
    circ.rz(qa, phi0)

    # Controlled-U on data (control = ancilla)
    add_token_pqc(circ, q0, q1, token_angles, control=qa)

    # Ry(phi1)
    circ.ry(qa, phi1)

    # Controlled-U† on data
    add_token_pqc_dagger(circ, q0, q1, token_angles, control=qa)

    # Rz(phi2)
    circ.rz(qa, phi2)

    if measure_all:
        circ.measure(q0, "d0")
        circ.measure(q1, "d1")
        circ.measure(qa, "anc")

    return circ


# --------------------------------------------
# 4. Helpers: run locally & compute <Z> values
# --------------------------------------------
def run_local(circuit, shots=4000):
    sim = LocalSimulator()
    task = sim.run(circuit, shots=shots)
    result = task.result()
    return result.measurement_counts


def z_expectation_from_counts(counts, bit_index):
    """
    <Z> = P(0) - P(1) on given bit position in the bitstring.
    Assumes bitstrings like 'd0d1ctrl' mapped in that order.
    """
    total = 0
    z = 0
    for bitstring, c in counts.items():
        bit = bitstring[bit_index]
        total += c
        z += c if bit == "0" else -c
    return z / total if total > 0 else 0.0


# -----------------------------
# 5. Example usage / comparison
# -----------------------------
if __name__ == "__main__":
    # Example token parameters
    token0_angles = {"a0": 0.3, "a1": -0.2, "b0": 0.5, "b1": -0.1}
    token1_angles = {"a0": -0.4, "a1": 0.6, "b0": -0.7, "b1": 0.2}
    gamma = 0.7
    encode_angles = {"x0": 0.2, "x1": -0.1}

    # For QSVT version, we just use one "token" U
    tokenU_angles = {"a0": 0.3, "a1": 0.5, "b0": -0.2, "b1": 0.4}
    qsvt_phis = (0.3, 0.9, -0.4)

    print("=== LCU-only Quixer-Mini ===")
    circ_lcu = build_quixer_mini_lcu(
        token0_angles,
        token1_angles,
        gamma,
        encode_angles=encode_angles,
    )
    print(circ_lcu)
    counts_lcu = run_local(circ_lcu, shots=4000)
    print("LCU counts:", counts_lcu)

    # Postselect on ctrl=0 (last bit)
    post_lcu = {b: c for b, c in counts_lcu.items() if b[-1] == "0"}
    z0_lcu = z_expectation_from_counts(post_lcu, 0)
    z1_lcu = z_expectation_from_counts(post_lcu, 1)
    print("<Z0>_LCU =", z0_lcu)
    print("<Z1>_LCU =", z1_lcu)
    print()

    print("=== QSVT-inspired Quixer-Mini ===")
    circ_qsvt = build_quixer_mini_with_qsvt(
        tokenU_angles,
        encode_angles=encode_angles,
        qsvt_phis=qsvt_phis,
    )
    print(circ_qsvt)
    counts_qsvt = run_local(circ_qsvt, shots=4000)
    print("QSVT counts:", counts_qsvt)

    # Here bit ordering is d0 (index 0), d1 (index 1), anc (index 2)
    z0_q = z_expectation_from_counts(counts_qsvt, 0)
    z1_q = z_expectation_from_counts(counts_qsvt, 1)
    print("<Z0>_QSVT =", z0_q)
    print("<Z1>_QSVT =", z1_q)
