# quixer_mini_v4.py
# ======================================================
# Quixer-Mini variants on AWS Braket LocalSimulator:
#   1) LCU-only block:      A = b0 U0 + b1 U1   (linear)
#   2) QSVT(U)-inspired:    nonlinear block on a single unitary U
#   3) QSVT(A)-inspired:    nonlinear block on full LCU gadget A
#
# Uses correct CRY decomposition and proper A / A† adjoints.
# Runs entirely on LocalSimulator (no AWS creds needed).

from braket.circuits import Circuit
from braket.devices import LocalSimulator
import numpy as np


# ---------------------------------------------------------
# 0. Utility: controlled-RY via standard CRY decomposition
# ---------------------------------------------------------
def cry(circ: Circuit, control: int, target: int, theta: float):
    """
    Apply a controlled-RY(theta) gate with given control and target
    using the standard decomposition:

        CRY(theta) = RY(theta/2) · CNOT · RY(-theta/2) · CNOT
    """
    half = theta / 2.0
    circ.ry(target, half)
    circ.cnot(control, target)
    circ.ry(target, -half)
    circ.cnot(control, target)
    return circ


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

    If control is not None, each RY is implemented as CRY(control -> target).
    """
    a0, a1 = angles["a0"], angles["a1"]
    b0, b1 = angles["b0"], angles["b1"]

    if control is None:
        circ.ry(q0, a0)
        circ.ry(q1, a1)
    else:
        cry(circ, control, q0, a0)
        cry(circ, control, q1, a1)

    # Entangling layer (uncontrolled)
    circ.cnot(q0, q1)

    if control is None:
        circ.ry(q0, b0)
        circ.ry(q1, b1)
    else:
        cry(circ, control, q0, b0)
        cry(circ, control, q1, b1)

    return circ


def add_token_pqc_dagger(circ, q0, q1, angles, control=None):
    """
    Adjoint of add_token_pqc.
    We reverse the order of gates and negate angles.
    """
    a0, a1 = angles["a0"], angles["a1"]
    b0, b1 = angles["b0"], angles["b1"]

    # Inverse of second-layer RY/CRY
    if control is None:
        circ.ry(q1, -b1)
        circ.ry(q0, -b0)
    else:
        cry(circ, control, q1, -b1)
        cry(circ, control, q0, -b0)

    # Inverse of CNOT (self-inverse)
    circ.cnot(q0, q1)

    # Inverse of first-layer RY/CRY
    if control is None:
        circ.ry(q1, -a1)
        circ.ry(q0, -a0)
    else:
        cry(circ, control, q1, -a1)
        cry(circ, control, q0, -a0)

    return circ


# -------------------------------------------------------
# 2. LCU "A" block and its adjoint: A ~ b0 U0 + b1 U1
# -------------------------------------------------------
def apply_lcu_A(circ, q0, q1, qc, gamma, token0_angles, token1_angles):
    """
    Coherent LCU block A on data qubits (q0,q1) with LCU control qc:

        A ~ b0 U0 + b1 U1,  with  b0 = cos(gamma), b1 = sin(gamma)

    SAME structure as the LCU-only circuit, but WITHOUT measurements
    or postselection. This can be reused inside QSVT-inspired blocks.
    """
    # Prepare b0|0> + b1|1> on qc
    circ.ry(qc, 2.0 * gamma)

    # Controlled U0 when qc = 0  (X sandwich)
    circ.x(qc)
    add_token_pqc(circ, q0, q1, token0_angles, control=qc)
    circ.x(qc)

    # Controlled U1 when qc = 1
    add_token_pqc(circ, q0, q1, token1_angles, control=qc)

    # Uncompute superposition
    circ.ry(qc, -2.0 * gamma)

    return circ


def apply_lcu_A_dagger(circ, q0, q1, qc, gamma, token0_angles, token1_angles):
    """
    Adjoint of apply_lcu_A.
    This is:

        A† = RY(+2γ)
             U1†
             X U0† X
             RY(-2γ)
    """
    # Inverse of final RY(-2γ) -> RY(+2γ)
    circ.ry(qc, 2.0 * gamma)

    # Inverse of "U1" block: controlled-U1† when qc = 1
    add_token_pqc_dagger(circ, q0, q1, token1_angles, control=qc)

    # Inverse of "X U0 X" block: X · U0†(qc=0) · X
    circ.x(qc)
    add_token_pqc_dagger(circ, q0, q1, token0_angles, control=qc)
    circ.x(qc)

    # Inverse of initial RY(2γ) -> RY(-2γ)
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
        circ.measure(q0)
        circ.measure(q1)
        circ.measure(qc)

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
        q2     : ancilla for QSVT-inspired block

    Use ONE token PQC U on (q0,q1) as "A", then apply a 1-step
    QSVT-inspired block with ancilla on q2:

        Rz(phi0) on anc
        controlled-U on data (control = anc)
        Ry(phi1) on anc
        controlled-U† on data (control = anc)
        Rz(phi2) on anc

    This implements a small polynomial-like transform p(U).
    (Not formal QSVT; just QSVT-inspired.)
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
        circ.measure(q0)
        circ.measure(q1)
        circ.measure(qa)

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
        q3     : ancilla for QSVT-inspired block

    QSVT-inspired structure (not formal QSVT):

        - Encode input on data
        - Apply A once (LCU block)
        - Rz(phi0) on ancilla
        - Apply A again
        - Ry(phi1) on ancilla
        - Apply A† (LCU dagger)
        - Rz(phi2) on ancilla

    This captures a "polynomial in A" flavor, still small enough
    for NISQ testing.
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

    # 1) Apply A once (linear attention-ish)
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
        circ.measure(q0)
        circ.measure(q1)
        circ.measure(qc)
        circ.measure(qa)

    return circ


# --------------------------------------------
# 6. Helpers: run locally & handle counts
# --------------------------------------------
def run_local(circuit, shots=4000):
    sim = LocalSimulator()
    task = sim.run(circuit, shots=shots)
    result = task.result()
    return result.measurement_counts


def flatten_counts(measurement_counts):
    """
    Normalize measurement_counts into {bitstring(str) -> count(int)}.

    Handles:
      - flat dict: {'000': 10, '011': 5}
      - nested dict: {('d0','d1'): {('0','1'): 7, ...}}
    """
    # Already flat (str -> int)?
    some_key = next(iter(measurement_counts))
    some_val = measurement_counts[some_key]

    if isinstance(some_key, str) and isinstance(some_val, int):
        return measurement_counts

    # Nested dict keyed by classical register names
    flat = {}
    for _, inner in measurement_counts.items():
        if isinstance(inner, dict):
            for bits, c in inner.items():
                # bits might be a tuple of '0'/'1'
                if isinstance(bits, tuple):
                    bitstring = "".join(bits)
                else:
                    bitstring = str(bits)
                flat[bitstring] = flat.get(bitstring, 0) + c
        else:
            # Fallback: treat outer key as bits
            key = some_key
            if isinstance(key, tuple):
                bitstring = "".join(key)
            else:
                bitstring = str(key)
            flat[bitstring] = flat.get(bitstring, 0) + some_val
    return flat


def z_expectation_from_counts(flat_counts, bit_index):
    """
    <Z> = P(0) - P(1) on given bit position in the bitstring.
    Assumes bitstring[bit_index] is '0'/'1'.
    """
    total = 0
    z = 0
    for bitstring, c in flat_counts.items():
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
    raw_lcu = run_local(circ_lcu, shots=4000)
    flat_lcu = flatten_counts(raw_lcu)
    print("LCU flat counts:", flat_lcu)

    # Postselect on lcu-control = 0 (bit 2 = '0')
    total_shots = sum(flat_lcu.values())
    post_lcu = {b: c for b, c in flat_lcu.items() if b[-1] == "0"}
    succ_shots = sum(post_lcu.values())
    p_success = succ_shots / total_shots if total_shots > 0 else 0.0
    print(f"LCU postselection success probability: {p_success:.3f}")

    z0_lcu = z_expectation_from_counts(post_lcu, 0)
    z1_lcu = z_expectation_from_counts(post_lcu, 1)
    print("<Z0>_LCU =", z0_lcu)
    print("<Z1>_LCU =", z1_lcu)
    print()

    # 2) QSVT-on-U (inspired)
    print("=== QSVT-on-U (inspired) ===")
    circ_qU = build_quixer_mini_with_qsvt_U(
        tokenU_angles,
        encode_angles=encode_angles,
        qsvt_phis=qsvt_phis,
    )
    print(circ_qU)
    raw_qU = run_local(circ_qU, shots=4000)
    flat_qU = flatten_counts(raw_qU)
    print("QSVT(U) flat counts:", flat_qU)
    z0_qU = z_expectation_from_counts(flat_qU, 0)
    z1_qU = z_expectation_from_counts(flat_qU, 1)
    print("<Z0>_QSVT(U) =", z0_qU)
    print("<Z1>_QSVT(U) =", z1_qU)
    print()

    # 3) QSVT-on-A (LCU, inspired)
    print("=== QSVT-on-A(LCU) (inspired) ===")
    circ_qA = build_quixer_mini_with_qsvt_full_lcu(
        token0_angles,
        token1_angles,
        gamma,
        encode_angles=encode_angles,
        qsvt_phis=qsvt_phis,
    )
    print(circ_qA)
    raw_qA = run_local(circ_qA, shots=4000)
    flat_qA = flatten_counts(raw_qA)
    print("QSVT(A) flat counts:", flat_qA)

    # Bit layout here: [d0, d1, lcu, anc] -> indices 0,1,2,3
    z0_qA = z_expectation_from_counts(flat_qA, 0)
    z1_qA = z_expectation_from_counts(flat_qA, 1)
    print("<Z0>_QSVT(A) =", z0_qA)
    print("<Z1>_QSVT(A) =", z1_qA)
