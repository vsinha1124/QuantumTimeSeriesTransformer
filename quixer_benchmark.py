# quixer_mini_v3.py
# ======================================================
# Quixer-Mini variants on AWS Braket LocalSimulator:
#   1) LCU-only block:      A = b0 U0 + b1 U1   (linear)
#   2) QSVT(U):             nonlinear block on a single token unitary U
#   3) QSVT(A=LCU):         nonlinear block on full LCU gadget A
#
# No AWS account needed; uses local simulator only.

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

    If control is not None, the RY gates are controlled by `control`.
    CNOT is left uncontrolled (shared structure for all tokens).
    """
    a0, a1 = angles["a0"], angles["a1"]
    b0, b1 = angles["b0"], angles["b1"]

    if control is None:
        circ.ry(q0, a0)
        circ.ry(q1, a1)
    else:
        circ.ry(q0, a0, control=control)
        circ.ry(q1, a1, control=control)

    # Simple entangling layer
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
    Adjoint of add_token_pqc.
    Reverse order of gates, negate rotation angles.
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


# -------------------------------------------------------
# 2. LCU "A" block and its adjoint: A ~ b0 U0 + b1 U1
# -------------------------------------------------------
def apply_lcu_A(circ, q0, q1, qc, gamma, token0_angles, token1_angles):
    """
    Coherent LCU block A on data qubits (q0,q1) with LCU control qc:

        A ~ b0 U0 + b1 U1,  with  b0 = cos(gamma), b1 = sin(gamma)

    This is the SAME structure as the LCU-only circuit, but WITHOUT
    measurements or postselection. It can be reused inside QSVT blocks.
    """
    # Prepare b0|0> + b1|1> on qc
    circ.ry(qc, 2.0 * gamma)

    # Controlled U0 when qc = 0 (X sandwich trick)
    circ.x(qc)
    add_token_pqc(circ, q0, q1, token0_angles, control=qc)
    circ.x(qc)

    # Controlled U1 when qc = 1
    add_token_pqc(circ, q0, q1, token1_angles, control=qc)

    # Uncompute qc superposition
    circ.ry(qc, -2.0 * gamma)

    return circ


def apply_lcu_A_dagger(circ, q0, q1, qc, gamma, token0_angles, token1_angles):
    """
    Adjoint of apply_lcu_A: reverse the sequence and invert angles.
    """
    # Inverse of final RY(-2 gamma) -> RY(+2 gamma)
    circ.ry(qc, 2.0 * gamma)

    # Inverse of U1 block: controlled-U1† when qc = 1
    add_token_pqc_dagger(circ, q0, q1, token1_angles, control=qc)

    # Inverse of U0 block: controlled-U0† when qc = 0
    circ.x(qc)
    add_token_pqc_dagger(circ, q0, q1, token0_angles, control=qc)
    circ.x(qc)

    # Inverse of initial RY(2 gamma) -> RY(-2 gamma)
    circ.ry(qc, -2.0 * gamma)

    return circ


# ----------------------------------------------
# 3. Variant 1: LCU-only Quixer-Mini (linear A)
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
        q2     : LCU control

    Implements:
        A = b0 U0 + b1 U1   (linear combination of token unitaries)

    This is the *linear* block only (no QSVT).
    """
    circ = Circuit()
    q0, q1 = 0, 1
    qc = 2   # LCU control

    # Optional: encode classical input on data
    if encode_angles is not None:
        circ.ry(q0, encode_angles.get("x0", 0.0))
        circ.ry(q1, encode_angles.get("x1", 0.0))

    # LCU A
    apply_lcu_A(
        circ,
        q0=q0,
        q1=q1,
        qc=qc,
        gamma=gamma,
        token0_angles=token0_angles,
        token1_angles=token1_angles,
    )

    if measure_all:
        circ.measure(q0, "d0")
        circ.measure(q1, "d1")
        circ.measure(qc, "lcu")

    return circ


# ----------------------------------------------------------
# 4. Variant 2: QSVT-inspired block on a single unitary U
# ----------------------------------------------------------
def build_quixer_mini_with_qsvt_U(
    token_angles,
    encode_angles=None,
    qsvt_phis=(0.3, 0.9, -0.4),
    measure_all=True,
):
    """
    4-qubit circuit:

        q0, q1 : data
        q2     : ancilla for QSVT-style block

    Use ONE token PQC U on (q0,q1) as "A", then apply a 1-step
    QSVT-inspired block with ancilla on q2:

        Rz(phi0) on anc
        controlled-U on data (control = anc)
        Ry(phi1) on anc
        controlled-U† on data (control = anc)
        Rz(phi2) on anc

    This implements a small polynomial p(U) ~ a U + b U^2.
    """
    circ = Circuit()
    q0, q1 = 0, 1
    qa = 2

    phi0, phi1, phi2 = qsvt_phis

    # Optional: encode input
    if encode_angles is not None:
        circ.ry(q0, encode_angles.get("x0", 0.0))
        circ.ry(q1, encode_angles.get("x1", 0.0))

    # 1) Apply U once (linear mixing)
    add_token_pqc(circ, q0, q1, token_angles, control=None)

    # 2) QSVT-inspired ancilla block
    circ.rz(qa, phi0)
    add_token_pqc(circ, q0, q1, token_angles, control=qa)
    circ.ry(qa, phi1)
    add_token_pqc_dagger(circ, q0, q1, token_angles, control=qa)
    circ.rz(qa, phi2)

    if measure_all:
        circ.measure(q0, "d0")
        circ.measure(q1, "d1")
        circ.measure(qa, "anc")

    return circ


# -----------------------------------------------------------------
# 5. Variant 3: QSVT-inspired block on full LCU A = b0 U0 + b1 U1
# -----------------------------------------------------------------
def build_quixer_mini_with_qsvt_full_lcu(
    token0_angles,
    token1_angles,
    gamma,
    encode_angles=None,
    qsvt_phis=(0.3, 0.9, -0.4),
    measure_all=True,
):
    """
    5-qubit circuit:

        q0, q1 : data
        q2     : LCU control (for A = b0 U0 + b1 U1)
        q3     : ancilla for QSVT-style block

    QSVT-inspired structure (simplified):

        - Encode input on data
        - Apply A once (LCU block)
        - Rz(phi0) on ancilla
        - Apply A again
        - Ry(phi1) on ancilla
        - Apply A† (LCU dagger)
        - Rz(phi2) on ancilla

    For a fully rigorous QSVT, ancilla would strictly control A and A†;
    here we keep it "inspired" but still capture a polynomial-in-A flavor.
    """
    circ = Circuit()
    q0, q1 = 0, 1
    qc = 2   # LCU control
    qa = 3   # ancilla

    phi0, phi1, phi2 = qsvt_phis

    # Optional: encode input
    if encode_angles is not None:
        circ.ry(q0, encode_angles.get("x0", 0.0))
        circ.ry(q1, encode_angles.get("x1", 0.0))

    # 1) Apply A once (linear attention-like mixing)
    apply_lcu_A(
        circ,
        q0=q0,
        q1=q1,
        qc=qc,
        gamma=gamma,
        token0_angles=token0_angles,
        token1_angles=token1_angles,
    )

    # 2) QSVT-inspired ancilla block around A, A†
    circ.rz(qa, phi0)

    apply_lcu_A(
        circ,
        q0=q0,
        q1=q1,
        qc=qc,
        gamma=gamma,
        token0_angles=token0_angles,
        token1_angles=token1_angles,
    )

    circ.ry(qa, phi1)

    apply_lcu_A_dagger(
        circ,
        q0=q0,
        q1=q1,
        qc=qc,
        gamma=gamma,
        token0_angles=token0_angles,
        token1_angles=token1_angles,
    )

    circ.rz(qa, phi2)

    if measure_all:
        circ.measure(q0, "d0")
        circ.measure(q1, "d1")
        circ.measure(qc, "lcu")
        circ.measure(qa, "anc")

    return circ


# --------------------------------------------
# 6. Helpers: run locally & compute <Z> values
# --------------------------------------------
def run_local(circuit, shots=4000):
    sim = LocalSimulator()
    task = sim.run(circuit, shots=shots)
    result = task.result()
    return result.measurement_counts


def z_expectation_from_counts(counts, bit_index):
    """
    <Z> = P(0) - P(1) on given bit position in the bitstring.
    Assumes bitstrings like '000', '101', etc. in qubit index order.
    """
    total = 0
    z = 0
    for bitstring, c in counts.items():
        bit = bitstring[bit_index]
        total += c
        z += c if bit == "0" else -c
    return z / total if total > 0 else 0.0


# -----------------------------
# 7. Example usage / comparison
# -----------------------------
if __name__ == "__main__":
    # Example token parameters
    token0_angles = {"a0": 0.3,  "a1": -0.2, "b0": 0.5,  "b1": -0.1}
    token1_angles = {"a0": -0.4, "a1": 0.6,  "b0": -0.7, "b1": 0.2}
    gamma = 0.7
    encode_angles = {"x0": 0.2, "x1": -0.1}

    # For QSVT-on-U, use a single PQC U
    tokenU_angles = {"a0": 0.3, "a1": 0.5, "b0": -0.2, "b1": 0.4}
    qsvt_phis = (0.3, 0.9, -0.4)

    # 1) LCU-only
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

    # Postselect on lcu-control = 0 (last bit)
    post_lcu = {b: c for b, c in counts_lcu.items() if b[-1] == "0"}
    z0_lcu = z_expectation_from_counts(post_lcu, 0)
    z1_lcu = z_expectation_from_counts(post_lcu, 1)
    print("<Z0>_LCU =", z0_lcu)
    print("<Z1>_LCU =", z1_lcu)
    print()

    # 2) QSVT on single U
    print("=== QSVT-on-U Quixer-Mini ===")
    circ_qsvt_U = build_quixer_mini_with_qsvt_U(
        tokenU_angles,
        encode_angles=encode_angles,
        qsvt_phis=qsvt_phis,
    )
    print(circ_qsvt_U)
    counts_qsvt_U = run_local(circ_qsvt_U, shots=4000)
    print("QSVT(U) counts:", counts_qsvt_U)
    z0_qU = z_expectation_from_counts(counts_qsvt_U, 0)
    z1_qU = z_expectation_from_counts(counts_qsvt_U, 1)
    print("<Z0>_QSVT(U) =", z0_qU)
    print("<Z1>_QSVT(U) =", z1_qU)
    print()

    # 3) QSVT-inspired on full LCU A
    print("=== QSVT-on-A(LCU) Quixer-Mini ===")
    circ_qsvt_A = build_quixer_mini_with_qsvt_full_lcu(
        token0_angles,
        token1_angles,
        gamma,
        encode_angles=encode_angles,
        qsvt_phis=qsvt_phis,
    )
    print(circ_qsvt_A)
    counts_qsvt_A = run_local(circ_qsvt_A, shots=4000)
    print("QSVT(A) counts:", counts_qsvt_A)

    # Here bit ordering for QSVT(A) is:
    #   bit 0 -> d0, bit 1 -> d1, bit 2 -> lcu, bit 3 -> anc
    z0_qA = z_expectation_from_counts(counts_qsvt_A, 0)
    z1_qA = z_expectation_from_counts(counts_qsvt_A, 1)
    print("<Z0>_QSVT(A) =", z0_qA)
    print("<Z1>_QSVT(A) =", z1_qA)
