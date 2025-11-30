# quixer_architecture.py  (a.k.a. quixer_mini_v4 core circuits)
# ======================================================
# Quixer-Mini variants:
#   1) LCU-only block:      A = b0 U0 + b1 U1   (linear)
#   2) QSVT(U)-inspired:    nonlinear block on a single unitary U
#   3) QSVT(A)-inspired:    nonlinear block on full LCU gadget A
#
# This file ONLY defines circuits. It does NOT pick a device or run them.
# All execution happens in benchmark.py via DEVICE_MODE + get_device().

from braket.circuits import Circuit

# ---------------------------------------------------------
# 0. Utility: backwards-compatible measure helper
# ---------------------------------------------------------
def safe_measure(circ: Circuit, qubit: int, key: str):
    """
    Backwards-compatible measurement helper.

    - On modern Braket: uses circ.measure(qubit, key)
    - On older / different Circuit implementations: falls back to circ.measure(qubit)
    """
    try:
        circ.measure(qubit, key)
    except TypeError:
        circ.measure(qubit)
    return circ


# ---------------------------------------------------------
# 0.5 Utility: controlled-RY via standard CRY decomposition
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
        safe_measure(circ, q0, "d0")
        safe_measure(circ, q1, "d1")
        safe_measure(circ, qc, "lcu")

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
        safe_measure(circ, q0, "d0")
        safe_measure(circ, q1, "d1")
        safe_measure(circ, qa, "anc")

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
        safe_measure(circ, q0, "d0")
        safe_measure(circ, q1, "d1")
        safe_measure(circ, qc, "lcu")
        safe_measure(circ, qa, "anc")

    return circ


def pretty_circuit(circ: Circuit):
    """
    Robust pretty-printer for Braket circuits.

    Handles:
      - Operators with/without .parameters
      - dict parameters ({"angle": x})
      - list/tuple parameters ([x, y, z])
      - no parameters
      - single- or multi-qubit targets
    """
    print("----- Circuit (pretty) -----")

    for instr in circ.instructions:
        op = instr.operator
        opname = getattr(op, "name", op.__class__.__name__)

        # Normalize targets (target or targets depending on SDK)
        targets = getattr(instr, "target", getattr(instr, "targets", None))
        if isinstance(targets, int):
            targets = (targets,)
        elif targets is None:
            targets = ()
        tstr = ", ".join(str(q) for q in targets)

        # Safely get parameters if they exist
        params = getattr(op, "parameters", None)
        pstr = ""

        if isinstance(params, dict):
            if len(params) > 0:
                pstr = ", ".join(f"{k}={float(v):.4f}" for k, v in params.items())
        elif isinstance(params, (list, tuple)):
            if len(params) > 0:
                pstr = ", ".join(f"{float(v):.4f}" for v in params)
        elif params is not None:
            # Some odd non-None thing
            pstr = str(params)

        if pstr:
            print(f"{opname:12s} | targets: [{tstr}] | {pstr}")
        else:
            print(f"{opname:12s} | targets: [{tstr}]")

    print("-----------------------------\n")

def ascii_draw_circuit(circ: Circuit):
    """
    Lightweight ASCII drawer for Braket circuits.
    One row per qubit, fixed-width columns so things line up.

    It understands:
      - single-qubit gates (X, RY, RZ, etc.)
      - two-qubit gates (CNOT-style) as a vertical line with ● and X
      - measurement as 'M'
    """
    # 1. Figure out how many qubits we touch
    max_q = -1
    for instr in circ.instructions:
        targets = getattr(instr, "target", getattr(instr, "targets", None))
        if isinstance(targets, int):
            targets = (targets,)
        elif targets is None:
            targets = ()
        for t in targets:
            try:
                q = int(t)
                max_q = max(max_q, q)
            except Exception:
                pass
    if max_q < 0:
        print("(empty circuit)")
        return

    n_qubits = max_q + 1

    # 2. Each qubit gets a list of segments (columns). Each segment is width 3.
    lines = [[] for _ in range(n_qubits)]

    def add_column(col_segments):
        """
        col_segments: dict {qubit_index: segment_str_of_len_3}
        Others get a wire '───'.
        """
        for q in range(n_qubits):
            seg = col_segments.get(q, "───")
            # pad/truncate to width 3
            if len(seg) < 3:
                seg = seg.center(3)
            elif len(seg) > 3:
                seg = seg[:3]
            lines[q].append(seg)

    # 3. Walk instructions in order, build columns
    for instr in circ.instructions:
        op = instr.operator
        name = getattr(op, "name", op.__class__.__name__).upper()
        targets = getattr(instr, "target", getattr(instr, "targets", None))
        if isinstance(targets, int):
            targets = (targets,)
        elif targets is None:
            targets = ()

        # Two-qubit CNOT-like gates
        low_name = name.lower()
        if len(targets) == 2 and low_name in ("cnot", "cz", "xy", "xx", "yy", "zz"):
            c, t = targets[0], targets[1]
            c = int(c)
            t = int(t)
            lo, hi = min(c, t), max(c, t)
            col = {}
            for q in range(lo, hi + 1):
                if q == c:
                    col[q] = "─●─"
                elif q == t:
                    col[q] = "─X─"
                else:
                    col[q] = "─│─"
            add_column(col)
            continue

        # Measurement gates: show as M on their target
        if "MEAS" in name or "MEASURE" in name:
            col = {}
            for q in targets:
                q = int(q)
                col[q] = "─M─"
            add_column(col)
            continue

        # Single-qubit (or treated-as-separate) gates
        # We'll just put the 3-char op name on each target, and wires elsewhere.
        label = name
        if len(label) > 3:
            label = label[:3]
        col = {}
        for q in targets:
            q = int(q)
            col[q] = label
        add_column(col)

    # 4. Print result
    print("===== ASCII circuit =====")
    for q in range(n_qubits):
        wire = "".join(lines[q]) if lines[q] else ""
        print(f"q{q}: {wire}")
    print("=========================\n")


if __name__ == "__main__":
    # Quick demo parameters
    token0_angles = {"a0": 0.3,  "a1": -0.2, "b0": 0.5,  "b1": -0.1}
    token1_angles = {"a0": -0.4, "a1": 0.6,  "b0": -0.7, "b1": 0.2}
    tokenU_angles = {"a0": 0.3, "a1": 0.5, "b0": -0.2, "b1": 0.4}
    gamma = 0.7
    encode_angles = {"x0": 0.2, "x1": -0.1}
    qsvt_phis = (0.3, 0.9, -0.4)

    print("\nBuilding LCU-only Quixer-Mini circuit...\n")
    circ_lcu = build_quixer_mini_lcu(
        token0_angles, token1_angles, gamma,
        encode_angles=encode_angles, measure_all=True
    )
    # pretty_circuit(circ_lcu)
    # ascii_draw_circuit(circ_lcu)

    print("Building QSVT-on-U (inspired) circuit...\n")
    circ_qU = build_quixer_mini_with_qsvt_U(
        tokenU_angles,
        encode_angles=encode_angles,
        qsvt_phis=qsvt_phis,
        measure_all=True,
    )
    # pretty_circuit(circ_qU)
    # ascii_draw_circuit(circ_qU)

    print("Building QSVT-on-A (LCU) (inspired) circuit...\n")
    circ_qA = build_quixer_mini_with_qsvt_full_lcu(
        token0_angles, token1_angles, gamma,
        encode_angles=encode_angles,
        qsvt_phis=qsvt_phis,
        measure_all=True,
    )
    # pretty_circuit(circ_qA)
    # ascii_draw_circuit(circ_qA)

    try:
        from braket.devices import LocalSimulator
        sim = LocalSimulator()

        print("Running LCU circuit on LocalSimulator:")
        res = sim.run(circ_lcu, shots=200).result().measurement_counts
        print("Counts:", res, "\n")

        print("Running QSVT(U) circuit on LocalSimulator:")
        res = sim.run(circ_qU, shots=200).result().measurement_counts
        print("Counts:", res, "\n")

        print("Running QSVT(A) circuit on LocalSimulator:")
        res = sim.run(circ_qA, shots=200).result().measurement_counts
        print("Counts:", res, "\n")

    except Exception as e:
        print("LocalSimulator unavailable:", e)
