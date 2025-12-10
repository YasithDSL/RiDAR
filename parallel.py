from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGOpNode
from circuit_analysis import find_qubit_reuse_pairs, modify_circuit, last_index_operation, first_index_operation
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import stim
from qiskit import QuantumCircuit
import stim
import sys

def apply_caqr(qc: QuantumCircuit):
    cur_qc = qc.copy()
    reuse_pairs = find_qubit_reuse_pairs(cur_qc)
    iter_count = 0
    chain = []
    weight = (1, 1, 1)


    while len(reuse_pairs) > 0 and iter_count < len(qc.qubits) - 1:
        depth_diff = sys.maxsize    
        lst_index = last_index_operation(cur_qc)
        fst_index = first_index_operation(cur_qc)
        
        for i in range(len(reuse_pairs)):
            test_qc = cur_qc.copy()
            test_out_qc = modify_circuit(test_qc, reuse_pairs[i])

            if weight[0]*(test_out_qc.depth() - cur_qc.depth()) + weight[1]* lst_index[reuse_pairs[i][0]]+weight[2]*abs(lst_index[reuse_pairs[i][0]] - fst_index[reuse_pairs[i][1]]) < depth_diff:
                depth_diff = test_out_qc.depth() - cur_qc.depth() + 0.5*lst_index[reuse_pairs[i][0]]
                best_pair = reuse_pairs[i]

        chain.append((best_pair[0], best_pair[1]))
        modified_qc = modify_circuit(cur_qc, best_pair)

        reuse_pairs = find_qubit_reuse_pairs(modified_qc)
        cur_qc = modified_qc.copy()
        iter_count += 1
        lst_index = last_index_operation(cur_qc)

    return cur_qc, iter_count, chain

def test_bv_circuit(qc, secret_string, circuit_name="Circuit"):
    test_qc = qc.copy()

    simulator = AerSimulator()
    job = simulator.run(test_qc, shots=1000)
    result = job.result()
    counts = result.get_counts()
    
    print(f"\n--- Testing {circuit_name} ---")
    print(f"Expected secret string: {secret_string}")
    print(f"Measurement results:")
    
    most_common = max(counts, key=counts.get)
    
    n = len(secret_string)
    
    for key, count in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:5]:
        measured_bits = key[-n:] if len(key) >= n else key
        percentage = 100 * count / 1000
        match = "✓" if measured_bits == secret_string else "✗"
        print(f"  {measured_bits}: {count:4d} ({percentage:5.1f}%) {match}")
    
    result_bits = most_common[-n:] if len(most_common) >= n else most_common
    is_correct = result_bits == secret_string
    
    if is_correct:
        print(f"✓ SUCCESS: Circuit correctly identifies secret string!")
    else:
        print(f"✗ FAILED: Got {result_bits}, expected {secret_string}")
        
    return is_correct

def qiskit_to_stim_circuit_with_noise(
    qc: QuantumCircuit,
    p1: float = 0.0,
    p2: float = 0.0,
) -> stim.Circuit:
    """Convert a Qiskit circuit to a Stim circuit, optionally adding noise."""
    single_qubit_gate_map = {
        "h": "H",
        "x": "X",
        "y": "Y",
        "z": "Z",
        "s": "S",
        "sdg": "S_DAG",
        "sx": "SQRT_X",
        "measure": "MR",
        "reset": "R",
    }

    stim_circuit = stim.Circuit()

    for gate in qc:
        op_name = gate.operation.name.lower()
        qubit = qc.find_bit(gate.qubits[0])[0]

        if op_name in single_qubit_gate_map:
            stim_op = single_qubit_gate_map[op_name]
            stim_circuit.append(stim_op, [qubit])

            # Add 1-qubit depolarising noise *after* non-measure/non-reset gates
            if p1 > 0 and op_name not in ["measure", "reset"]:
                stim_circuit.append("DEPOLARIZE1", [qubit], p1)

        elif op_name == "cx":
            target = qc.find_bit(gate.qubits[1])[0]
            stim_circuit.append("CX", [qubit, target])

            if p2 > 0:
                stim_circuit.append("DEPOLARIZE2", [qubit, target], p2)

        elif op_name == "barrier":
            stim_circuit.append("TICK")

        else:
            raise ValueError(f"Unsupported gate: {op_name}")

    return stim_circuit

import numpy as np

def run_bv_on_stim(
    qc: QuantumCircuit,
    secret_string: str,
    p1: float = 0.0,
    p2: float = 0.0,
    shots: int = 20_000,
    label: str = "BV on Stim",
) -> float:   
    n = len(secret_string)

    stim_circuit = qiskit_to_stim_circuit_with_noise(qc, p1=p1, p2=p2)

    # 2) Build measurement -> classical-bit mapping
    #    meas_map[k] = classical_bit_index that Stim measurement k writes to
    meas_map = []
    meas_counter = 0

    clbit_index = {cb: idx for idx, cb in enumerate(qc.clbits)}

    for gate in qc:
        op_name = gate.operation.name.lower()
        if op_name == "measure":
            cbit = gate.clbits[0]
            c_idx = clbit_index[cbit]
            meas_map.append((meas_counter, c_idx))
            meas_counter += 1

    num_meas_stim = stim_circuit.num_measurements
    if num_meas_stim != meas_counter:
        print(f"WARNING: Qiskit saw {meas_counter} measures, Stim has {num_meas_stim}.")

    # 3) Sample from Stim
    sampler = stim_circuit.compile_sampler()
    samples = sampler.sample(shots)  # shape (shots, num_measurements)

    # 4) Reconstruct Qiskit-style bitstrings for the BV bits
    counts = {}

    for shot in samples:
        cvals = [0] * n

        for meas_idx, c_idx in meas_map:
            if c_idx < n:
                cvals[c_idx] = int(shot[meas_idx])

        bits = []
        for c_idx in range(n - 1, -1, -1):
            bits.append(str(cvals[c_idx]))
        bitstring = ''.join(bits)
        
        counts[bitstring] = counts.get(bitstring, 0) + 1

    success_prob = counts.get(secret_string, 0) / shots
    return success_prob

import pymatching

def estimate_pL_surface_code(distance: int,
                             rounds: int = 10,
                             p_phys: float = 1e-3,
                             shots: int = 50_000) -> float:
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_x",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=p_phys,
    )

    dem = circuit.detector_error_model(decompose_errors=True)
    matching = pymatching.Matching.from_detector_error_model(dem)

    sampler = circuit.compile_detector_sampler()
    syndromes, observables = sampler.sample(
        shots=shots,
        separate_observables=True,
    )

    predicted = matching.decode_batch(syndromes)
    failures = np.sum(np.any(predicted != observables, axis=1))
    p_fail_total = failures / shots
    p_L = p_fail_total / rounds
    return p_L

import stim

def phys_qubits_per_logical(distance: int) -> int:
    circ = stim.Circuit.generated(
        "surface_code:rotated_memory_x",
        distance=distance,
        rounds=1,
        after_clifford_depolarization=0.0,
    )
    return circ.num_qubits

from qiskit import QuantumCircuit

def spacetime_volume_for_circuit(
    qc: QuantumCircuit,
    distance: int,
    rounds_per_logical_layer: int | None = None,
) -> int:
    n_logical = qc.num_qubits  # for BV you can keep it as is (data + ancilla)
    depth = qc.depth()

    if rounds_per_logical_layer is None:
        rounds_per_logical_layer = distance  # simple O(d) time model

    Q_phys = phys_qubits_per_logical(distance) * n_logical
    T_cycles = depth * rounds_per_logical_layer

    return Q_phys * T_cycles

def spacetime_volume_for_circuit(
    qc: QuantumCircuit,
    distance: int,
    rounds_per_logical_layer: int | None = None,
) -> int:
    n_logical = qc.num_qubits  # for BV you can keep it as is (data + ancilla)
    depth = qc.depth()

    if rounds_per_logical_layer is None:
        rounds_per_logical_layer = distance  # simple O(d) time model

    Q_phys = phys_qubits_per_logical(distance) * n_logical
    T_cycles = depth * rounds_per_logical_layer

    return Q_phys * T_cycles

def build_bv_circuit(secret_string: str) -> QuantumCircuit:
    n = len(secret_string)
    qc = QuantumCircuit(n + 1, n)

    qc.x(n)
    qc.h(n)

    for i in range(n):
        qc.h(i)
        if secret_string[n - 1 - i] == '1':
            qc.cx(i, n)
        qc.h(i)

    for i in range(n):
        qc.measure(i, i)

    return qc

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

def build_xor_n_circuit(input_bits: str) -> QuantumCircuit:
    n = len(input_bits)
    assert n >= 1

    q = QuantumRegister(n + 1, "q")
    c_out = ClassicalRegister(1, "out")
    qc = QuantumCircuit(q, c_out)

    # Prepare classical input on q[0..n-1]
    for i, b in enumerate(input_bits):
        if b == "1":
            qc.x(q[i])

    # Ancilla q[n] starts in |0>, compute XOR onto it
    for i in range(n):
        qc.cx(q[i], q[n])

    # Measure ancilla
    qc.measure(q[n], c_out[0])

    return qc


def xor_parity(input_bits: str) -> str:
    """Expected XOR_n output ('0' or '1')."""
    return str(sum(int(b) for b in input_bits) % 2)

import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
from scipy.optimize import curve_fit

def logistic_model(x, L, k, x0, b):
    return L / (1 + np.exp(-k * (np.log10(x) - x0))) + b

def get_active_qubit_count(qc: QuantumCircuit) -> int:
    used_qubits = set()
    for instruction in qc.data:
        for q in instruction.qubits:
            used_qubits.add(q)
    return len(used_qubits)

def circuit_space_time_volume(qc) -> int:
    return get_active_qubit_count(qc) * qc.depth()

# def physical_space_time_volume(qc, distance: int, rounds_per_layer: int = None) -> int:
#     n_logical = get_active_qubit_count(qc) # otherwise reuse is not credited
#     depth = qc.depth()
    
#     if rounds_per_layer is None:
#         rounds_per_layer = distance
    
#     space = n_logical * (distance ** 2)
#     time = depth * rounds_per_layer
    
#     return space * time

def physical_space_time_volume(qc, distance: int, rounds_per_layer: int = None) -> int:
    n_logical = get_active_qubit_count(qc) # otherwise reuse is not credited
    depth = qc.depth()
    
    if rounds_per_layer is None:
        rounds_per_layer = distance
    
    space = n_logical * (distance ** 2)
    time = get_logical_circuit_time(qc)
    
    return space * time

from qiskit import QuantumCircuit

def build_ghz_circuit(n: int) -> QuantumCircuit:
    qc = QuantumCircuit(n, n)

    qc.h(0)

    for i in range(n - 1):
        qc.cx(i, i + 1)

    qc.measure(range(n), range(n))

    return qc

def run_ghz_on_stim(
    qc: QuantumCircuit,
    n: int,
    p1: float = 0.0,
    p2: float = 0.0,
    shots: int = 10_000,
) -> dict:
    stim_circuit = qiskit_to_stim_circuit_with_noise(qc, p1=p1, p2=p2)
    
    sampler = stim_circuit.compile_sampler()
    samples = sampler.sample(shots)
    
    all_zeros = '0' * n
    all_ones = '1' * n
    counts = {}
    
    for shot in samples:
        bitstring = ''.join(str(int(b)) for b in shot[:n][::-1])
        counts[bitstring] = counts.get(bitstring, 0) + 1
    
    p_all_zeros = counts.get(all_zeros, 0) / shots
    p_all_ones = counts.get(all_ones, 0) / shots
    p_other = 1.0 - p_all_zeros - p_all_ones
    
    is_valid = (p_all_zeros > 0.4 and p_all_ones > 0.4)

    return {
        'all_zeros_prob': p_all_zeros,
        'all_ones_prob': p_all_ones,
        'other_prob': p_other,
        'is_valid': is_valid,
        'counts': counts
    }

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

def build_simon_circuit(n: int, s: str) -> QuantumCircuit:
    if len(s) != n:
        raise ValueError(f"Secret string length {len(s)} must match n={n}")
    
    qr_in = QuantumRegister(n, 'x')
    qr_out = QuantumRegister(n, 'y')
    cr = ClassicalRegister(n, 'meas')
    
    qc = QuantumCircuit(qr_in, qr_out, cr)
    
    # 1. Superposition
    qc.h(qr_in)
    
    # 2. Oracle (built directly into circuit)
    s_rev = s[::-1]
    
    # Find pivot
    pivot = -1
    for i, bit in enumerate(s_rev):
        if bit == '1':
            pivot = i
            break
    
    # Build oracle gates
    for i in range(n):
        if pivot == -1:
            qc.cx(qr_in[i], qr_out[i])
            continue
            
        if i == pivot:
            continue
            
        qc.cx(qr_in[i], qr_out[i])
        
        if s_rev[i] == '1':
            qc.cx(qr_in[pivot], qr_out[i])
    
    # 3. Interference
    qc.h(qr_in)
    
    # 4. Measurement (only measure input register)
    qc.measure(qr_in, cr)
    
    return qc

def run_simon_on_stim(
    qc: QuantumCircuit,
    secret_s: str,
    p1: float = 0.0,
    p2: float = 0.0,
    shots: int = 20_000,
) -> float:
    n = len(secret_s)
    
    stim_circuit = qiskit_to_stim_circuit_with_noise(qc, p1=p1, p2=p2)
    
    # Build measurement mapping
    meas_map = []
    meas_counter = 0
    clbit_index = {cb: idx for idx, cb in enumerate(qc.clbits)}
    
    for gate in qc:
        op_name = gate.operation.name.lower()
        if op_name == "measure":
            cbit = gate.clbits[0]
            c_idx = clbit_index[cbit]
            meas_map.append((meas_counter, c_idx))
            meas_counter += 1
    
    # Sample from Stim
    sampler = stim_circuit.compile_sampler()
    samples = sampler.sample(shots)
    
    # Check orthogonality
    s_int = int(secret_s, 2)
    valid_count = 0
    
    for shot in samples:
        # Reconstruct measurement bitstring
        cvals = [0] * n
        for meas_idx, c_idx in meas_map:
            if c_idx < n:
                cvals[c_idx] = int(shot[meas_idx])
        
        # Convert to integer
        z_int = int(''.join(str(cvals[i]) for i in range(n-1, -1, -1)), 2)
        
        # Check orthogonality: z · s = 0 (mod 2)
        dot_product = bin(z_int & s_int).count('1') % 2
        
        if dot_product == 0:
            valid_count += 1
    
    fidelity = valid_count / shots
    return fidelity

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


def build_repetition_code_circuit(
    distance: int,
    logical_state: str = "0",
    inject_error_qubit: int | None = None,
) -> QuantumCircuit:
    if distance < 1:
        raise ValueError("distance must be >= 1")
    if logical_state not in {"0", "1"}:
        raise ValueError("logical_state must be '0' or '1'")
    if inject_error_qubit is not None and not (0 <= inject_error_qubit < distance):
        raise ValueError("inject_error_qubit must be in [0, distance-1]")

    qc = QuantumCircuit(distance, distance, name=f"repetition_d{distance}_{logical_state}")

    # 1) Prepare logical |0>_L or |1>_L on qubit 0
    if logical_state == "1":
        qc.x(0)

    # 2) Encode into distance physical qubits: |b> → |b b ... b>
    for j in range(1, distance):
        qc.cx(0, j)

    # 3) Optional: inject a single X error on one physical qubit
    if inject_error_qubit is not None:
        qc.x(inject_error_qubit)

    # qc.barrier()

    # 4) Measure all physical qubits
    qc.measure(range(distance), range(distance))

    return qc

def run_repetition_code_on_stim(
    qc: QuantumCircuit,
    distance: int,
    logical_state: str,
    p1: float = 0.0,
    p2: float = 0.0,
    shots: int = 20_000,
) -> float:
    stim_circuit = qiskit_to_stim_circuit_with_noise(qc, p1=p1, p2=p2)
    
    # Build measurement mapping
    meas_map = []
    meas_counter = 0
    clbit_index = {cb: idx for idx, cb in enumerate(qc.clbits)}
    
    for gate in qc:
        op_name = gate.operation.name.lower()
        if op_name == "measure":
            cbit = gate.clbits[0]
            c_idx = clbit_index[cbit]
            meas_map.append((meas_counter, c_idx))
            meas_counter += 1
    
    # Sample from Stim
    sampler = stim_circuit.compile_sampler()
    samples = sampler.sample(shots)
    
    # Decode with majority vote
    correct_count = 0
    expected_bit = int(logical_state)
    
    for shot in samples:
        # Reconstruct measurement bitstring
        cvals = [0] * distance
        for meas_idx, c_idx in meas_map:
            if c_idx < distance:
                cvals[c_idx] = int(shot[meas_idx])
        
        # Majority vote decoding
        ones_count = sum(cvals)
        decoded_bit = 1 if ones_count > distance // 2 else 0
        
        if decoded_bit == expected_bit:
            correct_count += 1
    
    fidelity = correct_count / shots
    return fidelity


##################################

def get_decoherence_fidelities(lifetimes, distance: int):
    fidelities = []
    T1 = 20000 #T1 time in SC is ~200µs and single qubit gate is ~10ns
    #https://www.spinquanta.com/news-detail/ultimate-guide-to-coherence-time?utm_source=chatgpt.com
    #https://postquantum.com/quantum-modalities/superconducting-qubits/?utm_source=chatgpt.com
    max_e_phys = 0.1 #derived from formula in Camps, Rrapaj, Klymko et al.
    
    for lifetime in lifetimes:
        #formula from Camps, Rrapaj, Klymko et al. 
        #Quantum Computing Technology Roadmaps and Capability Assessment for Scientific
        #Computing - An analysis of use cases from the NERSC workload
        e_phys = min(max_e_phys, 1-np.exp(-lifetime/T1))
        f_log = 1 - 0.1 * ((100*(e_phys))**((distance+1)/2))
        #maximum logical error should not surpass maximum physical error
        fidelities.append(max(f_log, 1-max_e_phys))

    # print("decoherence fidelities for each qubit life:\n", fidelities)
    return fidelities

def get_total_decoherence_fidelity(fidelities: list):
    total_fidelity = 1
    for f in fidelities:
        total_fidelity *= f
    return total_fidelity

from qiskit.converters import circuit_to_dag
import numpy as np

#one_beat is assumed to be all other gates
two_beat_gates = ['cx'] #we assume that the circuits are small enough that cnots do not require too much overhead
three_beat_gates = ['h']

def get_logical_circuit_time(qc: QuantumCircuit):
    """
    # TODO UPDATE THIS FUNCTION TO USE THE DAG SO DIFFERENT LENGTH OPS CAN STILL BE RAN AT THE SAME TIME
    
    we model the temporal circuit depth adhering to Table I from Kobori et al.'s LSQCA with 0s replaced with 1s

    we ignore preparation time at the beginning of the circuit, 
    but not after reset operations
    """
    dag = circuit_to_dag(qc)
    circuit_depth = 0
    for depth, layer in enumerate(dag.layers()):
        longest_time_op = 1
        
        for node in layer["graph"].op_nodes():
            #assume that a barrier cannot be at the same time as any other operation
            if node.name == 'barrier':
                longest_time_op = 0
                break
            elif node.name in two_beat_gates:
                longest_time_op = max(2, longest_time_op)
            elif node.name in three_beat_gates:
                longest_time_op = max(3, longest_time_op)

        circuit_depth += longest_time_op
    return circuit_depth

def get_average_lifetimes(qc: QuantumCircuit):
    dag = circuit_to_dag(qc)
    all_lifetimes = []
    dynamic_lifetimes = [0] * qc.num_qubits
    measured_qubits = []

    for depth, layer in enumerate(dag.layers()):
        longest_time_op = 1
        
        for node in layer["graph"].op_nodes():
            #assume that a barrier cannot be at the same time as any other operation
            if node.name == 'barrier':
                longest_time_op = 0
                break
            elif node.name in two_beat_gates:
                longest_time_op = max(2, longest_time_op)
            elif node.name in three_beat_gates:
                longest_time_op = max(3, longest_time_op)

            if node.name == 'measure':
                for q in node.qargs:
                    index = qc.qubits.index(q)
                    #fixes the qubit lifetime so it will not be incremented later, unless there is a reset
                    measured_qubits.append(index)
                    dynamic_lifetimes[index] += 1 #measurement takes 1 time unit

            if node.name == 'reset':
                for q in node.qargs:
                    index = qc.qubits.index(q)
                    all_lifetimes.append(dynamic_lifetimes[index])
                    dynamic_lifetimes[index] = 0
                    try:
                        measured_qubits.remove(index)
                    except:
                        ValueError()
        for i in range(qc.num_qubits):
            #only increment non-measured qubits, as measured ones no longer count as living
            if i not in measured_qubits:
                dynamic_lifetimes[i] += longest_time_op
        
    all_lifetimes += list(filter(lambda x: x != 0, dynamic_lifetimes))
    return {'avg_lifetime': np.mean(all_lifetimes), 'max_lifetime': np.max(all_lifetimes), 'lifetimes': all_lifetimes}

##################################

def analyze_fidelity_vs_space_time(
    string_lengths,
    distances,
    rounds: int = 10,
    p_phys_surface: float = 1e-2,
    shots_pL: int = 50_000,
    shots_algo: int = 20_000,
    circuit_type: str = "bv",  # "bv", "xor", "ghz", "simon", or "repetition"
):
    results = []
    
    circuit_name = {
        "bv": "BV", 
        "xor": "XOR_n", 
        "ghz": "GHZ", 
        "simon": "Simon",
        "repetition": "Repetition Code"
    }[circuit_type]

    for n in tqdm(string_lengths, desc="Circuit Sizes"):

        # 1) Build logical circuit based on type
        if circuit_type == "bv":
            secret = "1" * n 
            qc_seq = build_bv_circuit(secret)
            expected_outcome = secret
            
        elif circuit_type == "xor":
            xor_input = "1" * n 
            qc_seq = build_xor_n_circuit(xor_input)
            expected_outcome = xor_parity(xor_input)
            
        elif circuit_type == "ghz":
            qc_seq = build_ghz_circuit(n)
            expected_outcome = None
            
        elif circuit_type == "simon":
            secret = "1" + "01" * ((n-1)//2) + ("0" if n % 2 == 0 else "")
            secret = secret[:n]
            qc_seq = build_simon_circuit(n, secret)
            expected_outcome = secret
            
        elif circuit_type == "repetition":
            # For repetition code, n is the code distance
            logical_state = "1"  # Test with logical |1⟩
            qc_seq = build_repetition_code_circuit(
                distance=n,
                logical_state=logical_state,
                inject_error_qubit=None
            )
            expected_outcome = logical_state
            code_distance_for_rep = n
            
        else:
            raise ValueError(f"Unknown circuit_type: {circuit_type}")

        # 2) Apply CAQR
        qc_caqr, iter_count, _ = apply_caqr(qc_seq)

        # 3) Scan code distances
        for d in tqdm(distances, desc=f"Distances (n={n})", leave=False):

            pL = estimate_pL_surface_code(
                distance=d,
                rounds=rounds,
                p_phys=p_phys_surface,
                shots=shots_pL,
            )

            p1 = pL
            p2 = 2 * pL

            # 3a) Compute fidelity for original circuit
            if circuit_type == "ghz":
                result_orig = run_ghz_on_stim(qc_seq, n, p1=p1, p2=p2, 
                                           shots=shots_algo)
                F_orig = result_orig['all_zeros_prob'] + result_orig['all_ones_prob']
                
            elif circuit_type == "simon":
                F_orig = run_simon_on_stim(qc_seq, expected_outcome, p1=p1, p2=p2,
                                          shots=shots_algo)
                
            elif circuit_type == "repetition":
                F_orig = run_repetition_code_on_stim(
                    qc_seq, code_distance_for_rep, expected_outcome, 
                    p1=p1, p2=p2, shots=shots_algo
                )
                
            else:  # BV or XOR
                F_orig = run_bv_on_stim(qc_seq, expected_outcome, p1=p1, p2=p2,
                                       shots=shots_algo)

            # 3b) Compute fidelity for CAQR circuit
            if circuit_type == "ghz":
                result_caqr = run_ghz_on_stim(qc_caqr, n, p1=p1, p2=p2,
                                            shots=shots_algo)
                F_caqr = result_caqr['all_zeros_prob'] + result_caqr['all_ones_prob']
                
            elif circuit_type == "simon":
                F_caqr = run_simon_on_stim(qc_caqr, expected_outcome, p1=p1, p2=p2,
                                          shots=shots_algo)
                
            elif circuit_type == "repetition":
                F_caqr = run_repetition_code_on_stim(
                    qc_caqr, code_distance_for_rep, expected_outcome,
                    p1=p1, p2=p2, shots=shots_algo
                )
                
            else:  # BV or XOR
                F_caqr = run_bv_on_stim(qc_caqr, expected_outcome, p1=p1, p2=p2,
                                       shots=shots_algo)

            # 3c) Approximate logical space–time volume
            V_orig = physical_space_time_volume(qc_seq, distance=d)
            V_caqr = physical_space_time_volume(qc_caqr, distance=d)

            lifetimes_pre_caqr = get_average_lifetimes(qc_seq)['lifetimes']
            lifetimes_post_caqr = get_average_lifetimes(qc_caqr)['lifetimes']
            fidelity_decoherence_orig = get_total_decoherence_fidelity(
                get_decoherence_fidelities(lifetimes_pre_caqr, distance=d)
            )
            fidelity_decoherence_caqr = get_total_decoherence_fidelity(
                get_decoherence_fidelities(lifetimes_post_caqr, distance=d)
            )
            results.append({
                "n": n,
                "d": d,
                "reuses": iter_count,
                "V_orig": V_orig,
                "V_caqr": V_caqr,
                "F_orig": F_orig * fidelity_decoherence_orig,
                "F_caqr": F_caqr * fidelity_decoherence_caqr,
            })

    # 4) Plot fidelity vs space–time volume (AVERAGES ONLY)
    fig, ax = plt.subplots(figsize=(8, 6))

    # Group by distance and compute averages
    orig_by_d = defaultdict(lambda: {'V': [], 'F': []})
    caqr_by_d = defaultdict(lambda: {'V': [], 'F': []})
    
    for r in results:
        d = r['d']
        orig_by_d[d]['V'].append(r['V_orig'])
        orig_by_d[d]['F'].append(r['F_orig'])
        caqr_by_d[d]['V'].append(r['V_caqr'])
        caqr_by_d[d]['F'].append(r['F_caqr'])
    
    # Compute averages per distance
    V_o_avg = []
    F_o_avg = []
    V_c_avg = []
    F_c_avg = []
    distances_used = sorted(orig_by_d.keys())
    
    for d in distances_used:
        V_o_avg.append(np.mean(orig_by_d[d]['V']))
        F_o_avg.append(np.mean(orig_by_d[d]['F']))
        V_c_avg.append(np.mean(caqr_by_d[d]['V']))
        F_c_avg.append(np.mean(caqr_by_d[d]['F']))
    
    V_o_avg = np.array(V_o_avg)
    F_o_avg = np.array(F_o_avg)
    V_c_avg = np.array(V_c_avg)
    F_c_avg = np.array(F_c_avg)
    
    ax.set_xscale("log")
    # ax.set_yscale("log")

    # Plot only the averages per distance (no scatter, no fit)
    ax.plot(V_o_avg, F_o_avg, 'o-', color='blue', linewidth=2, markersize=8, 
            label="Original (avg)", alpha=0.8)
    ax.plot(V_c_avg, F_c_avg, 's-', color='orange', linewidth=2, markersize=8, 
            label="CAQR (avg)", alpha=0.8)
    
    # Optional: Add labels for each distance
    for i, d in enumerate(distances_used):
        ax.annotate(f'd={d}', (V_o_avg[i], F_o_avg[i]), 
                   fontsize=8, alpha=0.6, xytext=(5, 5), 
                   textcoords='offset points')
        ax.annotate(f'd={d}', (V_c_avg[i], F_c_avg[i]), 
                   fontsize=8, alpha=0.6, xytext=(5, -10), 
                   textcoords='offset points')

    ax.set_xlabel("Physical space–time volume", fontsize=12)
    ax.set_ylabel("Fidelity", fontsize=12)
    ax.set_title(f"{circuit_name}: Fidelity vs space–time volume (averages)", fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(f'{circuit_type}_fidelity_vs_volume.png', dpi=150, bbox_inches='tight')
    plt.close()  # Close instead of show for parallel execution
    
    print(f"Plot saved to {circuit_type}_fidelity_vs_volume.png")
    
    return results

import sys

# circuit_type = sys.argv[1]  # "bv", "xor", "ghz", "simon", or "repetition"

# distances = [n for n in range(3, 21, 2)]
# lengths = [n for n in range(3, 31, 4)]
# rounds = 10
# p_phys_surface = 1e-2
# shots_pL = 50_000
# shots_algo = 20_000

# print(f"\n{'='*90}")
# print(f"FIDELITY vs SPACE-TIME VOLUME - {circuit_type.upper()}")
# print(f"{'='*90}")

# results = analyze_fidelity_vs_space_time(
#     string_lengths=lengths,
#     distances=distances,
#     rounds=rounds,
#     p_phys_surface=p_phys_surface,
#     shots_pL=shots_pL,
#     shots_algo=shots_algo,
#     circuit_type=circuit_type,
# )

# # Save results
# import pickle
# with open(f'{circuit_type}_results.pkl', 'wb') as f:
#     pickle.dump(results, f)


def get_average_lifetimes(qc: QuantumCircuit) -> dict:
    """
    Compute qubit lifetimes accounting for resets.
    Returns dict with avg_lifetime, max_lifetime, and all lifetimes.
    """
    dag = circuit_to_dag(qc)
    all_lifetimes = []
    dynamic_lifetimes = [0] * qc.num_qubits
    measured_qubits = []

    for depth, layer in enumerate(dag.layers()):
        longest_time_op = 1

        for node in layer["graph"].op_nodes():
            if node.name == 'barrier':
                longest_time_op = 0
                break
            elif node.name in two_beat_gates:
                longest_time_op = max(2, longest_time_op)
            elif node.name in three_beat_gates:
                longest_time_op = max(3, longest_time_op)

            if node.name == 'measure':
                for q in node.qargs:
                    index = qc.qubits.index(q)
                    measured_qubits.append(index)
                    dynamic_lifetimes[index] += 1

            if node.name == 'reset':
                for q in node.qargs:
                    index = qc.qubits.index(q)
                    all_lifetimes.append(dynamic_lifetimes[index])
                    dynamic_lifetimes[index] = 0
                    if index in measured_qubits:
                        measured_qubits.remove(index)

        for i in range(qc.num_qubits):
            if i not in measured_qubits:
                dynamic_lifetimes[i] += longest_time_op

    all_lifetimes += list(filter(lambda x: x != 0, dynamic_lifetimes))
    
    if len(all_lifetimes) == 0:
        return {'avg_lifetime': 0, 'max_lifetime': 0, 'max_avg_ratio': 1.0, 'lifetimes': []}
    
    avg = np.mean(all_lifetimes)
    # max_lt = np.max(all_lifetimes)
    max_lt = get_logical_circuit_time(qc)
    ratio = max_lt / avg if avg > 0 else 1.0
    
    return {
        'avg_lifetime': avg, 
        'max_lifetime': max_lt, 
        'max_avg_ratio': ratio,
        'lifetimes': all_lifetimes
    }


def find_min_distance_for_fidelity(
    qc: QuantumCircuit,
    expected_outcome,
    circuit_type: str,
    target_fidelity: float = 0.90,
    max_distance: int = 21,
    rounds: int = 10,
    p_phys_surface: float = 1e-2,
    shots_pL: int = 50_000,
    shots_algo: int = 20_000,
    n: int = None,  # For GHZ/repetition
) -> int:
    for d in range(3, max_distance + 1, 2):  # Only odd distances
        pL = estimate_pL_surface_code(
            distance=d,
            rounds=rounds,
            p_phys=p_phys_surface,
            shots=shots_pL,
        )
        p1 = pL
        p2 = 2 * pL
        
        # Compute fidelity based on circuit type
        if circuit_type == "ghz":
            result = run_ghz_on_stim(qc, n, p1=p1, p2=p2, shots=shots_algo)
            fidelity = result['all_zeros_prob'] + result['all_ones_prob']
        elif circuit_type == "simon":
            fidelity = run_simon_on_stim(qc, expected_outcome, p1=p1, p2=p2, shots=shots_algo)
        elif circuit_type == "repetition":
            fidelity = run_repetition_code_on_stim(qc, n, expected_outcome, p1=p1, p2=p2, shots=shots_algo)
        else:  # BV or XOR
            fidelity = run_bv_on_stim(qc, expected_outcome, p1=p1, p2=p2, shots=shots_algo)
        
        decoherence = get_total_decoherence_fidelity(
            get_decoherence_fidelities(get_average_lifetimes(qc)['lifetimes'], distance=d)
        )
        print(decoherence)
        fidelity *= decoherence

        if fidelity >= target_fidelity:
            return d
    
    return max_distance + 2  # Could not achieve target


def analyze_min_distance_vs_lifetime_ratio(
    string_lengths,
    target_fidelity: float = 0.90,
    max_distance: int = 21,
    rounds: int = 10,
    p_phys_surface: float = 1e-2,
    shots_pL: int = 50_000,
    shots_algo: int = 20_000,
    circuit_type: str = "bv",
):
    results = []
    
    circuit_name = {
        "bv": "BV", 
        "xor": "XOR_n", 
        "ghz": "GHZ", 
        "simon": "Simon",
        "repetition": "Repetition Code"
    }[circuit_type]

    for n in tqdm(string_lengths, desc=f"Analyzing {circuit_name}"):
        
        # 1) Build circuit based on type
        if circuit_type == "bv":
            secret = "1" * n 
            qc_seq = build_bv_circuit(secret)
            expected_outcome = secret
        elif circuit_type == "xor":
            xor_input = "1" * n 
            qc_seq = build_xor_n_circuit(xor_input)
            expected_outcome = xor_parity(xor_input)
        elif circuit_type == "ghz":
            qc_seq = build_ghz_circuit(n)
            expected_outcome = None
        elif circuit_type == "simon":
            secret = "1" + "01" * ((n-1)//2) + ("0" if n % 2 == 0 else "")
            secret = secret[:n]
            qc_seq = build_simon_circuit(n, secret)
            expected_outcome = secret
        elif circuit_type == "repetition":
            logical_state = "1"
            qc_seq = build_repetition_code_circuit(distance=n, logical_state=logical_state)
            expected_outcome = logical_state
        else:
            raise ValueError(f"Unknown circuit_type: {circuit_type}")

        # 2) Apply CAQR
        qc_caqr, iter_count, chain = apply_caqr(qc_seq)
        
        # 3) Compute lifetime metrics for both circuits
        lifetimes_orig = get_average_lifetimes(qc_seq)
        lifetimes_caqr = get_average_lifetimes(qc_caqr)
        
        print(f"\nn={n}: Original ratio={lifetimes_orig['max_avg_ratio']:.2f}, "
              f"CAQR ratio={lifetimes_caqr['max_avg_ratio']:.2f}, reuses={iter_count}")
        
        # 4) Find minimum distance for original circuit
        print(f"  Finding min distance for original...", end=" ")
        min_d_orig = find_min_distance_for_fidelity(
            qc_seq, expected_outcome, circuit_type,
            target_fidelity=target_fidelity,
            max_distance=max_distance,
            rounds=rounds,
            p_phys_surface=p_phys_surface,
            shots_pL=shots_pL,
            shots_algo=shots_algo,
            n=n,
        )
        print(f"d={min_d_orig}")
        
        # 5) Find minimum distance for CAQR circuit
        print(f"  Finding min distance for CAQR...", end=" ")
        min_d_caqr = find_min_distance_for_fidelity(
            qc_caqr, expected_outcome, circuit_type,
            target_fidelity=target_fidelity,
            max_distance=max_distance,
            rounds=rounds,
            p_phys_surface=p_phys_surface,
            shots_pL=shots_pL,
            shots_algo=shots_algo,
            n=n,
        )
        print(f"d={min_d_caqr}")
        
        results.append({
            'n': n,
            'reuses': iter_count,
            'ratio_orig': lifetimes_orig['max_avg_ratio'],
            'ratio_caqr': lifetimes_caqr['max_avg_ratio'],
            'avg_lifetime_orig': lifetimes_orig['avg_lifetime'],
            'avg_lifetime_caqr': lifetimes_caqr['avg_lifetime'],
            'max_lifetime_orig': lifetimes_orig['max_lifetime'],
            'max_lifetime_caqr': lifetimes_caqr['max_lifetime'],
            'min_d_orig': min_d_orig,
             'min_d_caqr': min_d_caqr,
            'distance_reduction': min_d_orig - min_d_caqr,
        })

    # 5) Create plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    ratios_orig = [r['ratio_orig'] for r in results]
    ratios_caqr = [r['ratio_caqr'] for r in results]
    min_d_orig = [r['min_d_orig'] for r in results]
    min_d_caqr = [r['min_d_caqr'] for r in results]
    ns = [r['n'] for r in results]
    distance_reductions = [r['distance_reduction'] for r in results]
    
    # Filter out points where distance couldn't be achieved
    valid_orig = [(r, d, n) for r, d, n in zip(ratios_orig, min_d_orig, ns) if d <= max_distance]
    valid_caqr = [(r, d, n) for r, d, n in zip(ratios_caqr, min_d_caqr, ns) if d <= max_distance]
    
    # Plot 1: Min Distance vs Max/Avg Ratio (Original)
    if valid_orig:
        r_o, d_o, n_o = zip(*valid_orig)
        scatter1 = axes[0, 0].scatter(r_o, d_o, s=100, alpha=0.7, c=n_o, cmap='viridis')
        plt.colorbar(scatter1, ax=axes[0, 0], label='Circuit size (n)')
        for i, (r, d, n) in enumerate(valid_orig):
            axes[0, 0].annotate(f'n={n}', (r, d), fontsize=8, alpha=0.7, xytext=(3, 3), textcoords='offset points')
    axes[0, 0].set_xlabel('Max/Avg Lifetime Ratio', fontsize=12)
    axes[0, 0].set_ylabel(f'Min Distance for {target_fidelity:.0%} Fidelity', fontsize=12)
    axes[0, 0].set_title(f'{circuit_name} Original: Distance vs Lifetime Ratio', fontsize=14)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Min Distance vs Max/Avg Ratio (CAQR)
    if valid_caqr:
        r_c, d_c, n_c = zip(*valid_caqr)
        scatter2 = axes[0, 1].scatter(r_c, d_c, s=100, alpha=0.7, c=n_c, cmap='plasma')
        plt.colorbar(scatter2, ax=axes[0, 1], label='Circuit size (n)')
        for i, (r, d, n) in enumerate(valid_caqr):
            axes[0, 1].annotate(f'n={n}', (r, d), fontsize=8, alpha=0.7, xytext=(3, 3), textcoords='offset points')
    axes[0, 1].set_xlabel('Max/Avg Lifetime Ratio', fontsize=12)
    axes[0, 1].set_ylabel(f'Min Distance for {target_fidelity:.0%} Fidelity', fontsize=12)
    axes[0, 1].set_title(f'{circuit_name} CAQR: Distance vs Lifetime Ratio', fontsize=14)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Compare Original vs CAQR min distance
    axes[1, 0].plot(ns, min_d_orig, 'o-', label='Original', linewidth=2, markersize=8, color='blue')
    axes[1, 0].plot(ns, min_d_caqr, 's-', label='CAQR', linewidth=2, markersize=8, color='orange')
    axes[1, 0].axhline(y=max_distance, color='red', linestyle='--', alpha=0.5, label=f'Max tested (d={max_distance})')
    axes[1, 0].set_xlabel('Circuit Size (n)', fontsize=12)
    axes[1, 0].set_ylabel(f'Min Distance for {target_fidelity:.0%} Fidelity', fontsize=12)
    axes[1, 0].set_title(f'{circuit_name}: Distance Requirements', fontsize=14)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Distance Reduction vs CAQR Lifetime Ratio
    valid_reduction = [(r, dr, n) for r, dr, n in zip(ratios_caqr, distance_reductions, ns) 
                       if min_d_orig[ns.index(n)] <= max_distance and min_d_caqr[ns.index(n)] <= max_distance]
    if valid_reduction:
        r_red, d_red, n_red = zip(*valid_reduction)
        scatter4 = axes[1, 1].scatter(r_red, d_red, s=100, alpha=0.7, c=n_red, cmap='coolwarm')
        plt.colorbar(scatter4, ax=axes[1, 1], label='Circuit size (n)')
        for r, dr, n in valid_reduction:
            axes[1, 1].annotate(f'n={n}', (r, dr), fontsize=8, alpha=0.7, xytext=(3, 3), textcoords='offset points')
    axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('CAQR Max/Avg Lifetime Ratio', fontsize=12)
    axes[1, 1].set_ylabel('Distance Reduction (Original - CAQR)', fontsize=12)
    axes[1, 1].set_title(f'{circuit_name}: Distance Savings vs Lifetime Ratio', fontsize=14)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{circuit_type}_distance_vs_lifetime.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlot saved to {circuit_type}_distance_vs_lifetime.png")
    
    # Print summary table
    print("\n" + "="*100)
    print(f"{circuit_name}: MIN DISTANCE FOR {target_fidelity:.0%} FIDELITY vs LIFETIME RATIO")
    print("="*100)
    print(f"{'n':>3} | {'Reuses':>6} | {'Ratio_O':>8} | {'Ratio_C':>8} | {'d_orig':>6} | {'d_caqr':>6} | {'Δd':>4}")
    print("-"*100)
    for r in results:
        d_orig_str = str(r['min_d_orig']) if r['min_d_orig'] <= max_distance else ">max"
        d_caqr_str = str(r['min_d_caqr']) if r['min_d_caqr'] <= max_distance else ">max"
        print(f"{r['n']:3d} | {r['reuses']:6d} | {r['ratio_orig']:8.2f} | {r['ratio_caqr']:8.2f} | "
              f"{d_orig_str:>6} | {d_caqr_str:>6} | {r['distance_reduction']:4d}")
    
    return results


def analyze_combined_distance_vs_lifetime_ratio(
    target_fidelity: float = 0.90,
    max_distance: int = 21,
    rounds: int = 10,
    p_phys_surface: float = 1e-2,
    shots_pL: int = 50_000,
    shots_algo: int = 20_000,
):
    all_results = {
        "bv": [],
        "xor": [],
        "ghz": [],
        "simon": [],
        "repetition": []
    }
    
    circuit_configs = {
        "bv": [n for n in range(3, 15, 2)],
        "xor": [n for n in range(3, 15, 2)],
        "ghz": [n for n in range(3, 15, 2)],
        "simon": [n for n in range(3, 15, 2)],
        "repetition": [n for n in range(3, 15, 2)],
    }
    
    print("="*100)
    print("COMBINED ANALYSIS: ALL CIRCUIT TYPES")
    print("="*100)
    
    # Analyze each circuit type
    for circuit_type, string_lengths in circuit_configs.items():
        print(f"\n{'='*100}")
        print(f"Analyzing {circuit_type.upper()}")
        print(f"{'='*100}")
        
        for n in tqdm(string_lengths, desc=f"{circuit_type}"):
            
            # 1) Build circuit
            if circuit_type == "bv":
                secret = "1" * n 
                qc_seq = build_bv_circuit(secret)
                expected_outcome = secret
            elif circuit_type == "xor":
                xor_input = "1" * n 
                qc_seq = build_xor_n_circuit(xor_input)
                expected_outcome = xor_parity(xor_input)
            elif circuit_type == "ghz":
                qc_seq = build_ghz_circuit(n)
                expected_outcome = None
            elif circuit_type == "simon":
                secret = "1" + "01" * ((n-1)//2) + ("0" if n % 2 == 0 else "")
                secret = secret[:n]
                qc_seq = build_simon_circuit(n, secret)
                expected_outcome = secret
            elif circuit_type == "repetition":
                logical_state = "1"
                qc_seq = build_repetition_code_circuit(distance=n, logical_state=logical_state)
                expected_outcome = logical_state
            
            # 2) Apply CAQR
            qc_caqr, iter_count, _ = apply_caqr(qc_seq)
            
            # 3) Compute lifetime metrics
            lifetimes_orig = get_average_lifetimes(qc_seq)
            lifetimes_caqr = get_average_lifetimes(qc_caqr)
            
            # 4) Find minimum distances
            min_d_orig = find_min_distance_for_fidelity(
                qc_seq, expected_outcome, circuit_type,
                target_fidelity=target_fidelity,
                max_distance=max_distance,
                rounds=rounds,
                p_phys_surface=p_phys_surface,
                shots_pL=shots_pL,
                shots_algo=shots_algo,
                n=n,
            )
            
            min_d_caqr = find_min_distance_for_fidelity(
                qc_caqr, expected_outcome, circuit_type,
                target_fidelity=target_fidelity,
                max_distance=max_distance,
                rounds=rounds,
                p_phys_surface=p_phys_surface,
                shots_pL=shots_pL,
                shots_algo=shots_algo,
                n=n,
            )
            
            print(f"  {circuit_type} n={n}: ratio_orig={lifetimes_orig['max_avg_ratio']:.2f}, "
                  f"ratio_caqr={lifetimes_caqr['max_avg_ratio']:.2f}, "
                  f"d_orig={min_d_orig}, d_caqr={min_d_caqr}")
            
            all_results[circuit_type].append({
                'circuit_type': circuit_type,
                'n': n,
                'ratio_orig': lifetimes_orig['max_avg_ratio'],
                'ratio_caqr': lifetimes_caqr['max_avg_ratio'],
                'min_d_orig': min_d_orig,
                'min_d_caqr': min_d_caqr,
                'reuses': iter_count,
            })
    
    # Flatten results for plotting
    all_orig_points = []
    all_caqr_points = []
    
    for circuit_type, results in all_results.items():
        for r in results:
            if r['min_d_orig'] <= max_distance:
                all_orig_points.append({
                    'ratio': r['ratio_orig'],
                    'distance': r['min_d_orig'],
                    'circuit': circuit_type,
                    'n': r['n']
                })
            if r['min_d_caqr'] <= max_distance:
                all_caqr_points.append({
                    'ratio': r['ratio_caqr'],
                    'distance': r['min_d_caqr'],
                    'circuit': circuit_type,
                    'n': r['n']
                })
    
    # Create combined plots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Color map for different circuit types
    circuit_colors = {
        'bv': 'blue',
        'xor': 'green',
        'ghz': 'red',
        'simon': 'purple',
        'repetition': 'orange'
    }
    
    circuit_markers = {
        'bv': 'o',
        'xor': 's',
        'ghz': '^',
        'simon': 'D',
        'repetition': 'v'
    }
    
    circuit_names = {
        'bv': 'BV',
        'xor': 'XOR_n',
        'ghz': 'GHZ',
        'simon': 'Simon',
        'repetition': 'Rep. Code'
    }
    
    # Plot 1: Original circuits - all algorithms combined
    for circuit_type in ['bv', 'xor', 'ghz', 'simon', 'repetition']:
        points = [p for p in all_orig_points if p['circuit'] == circuit_type]
        if points:
            ratios = [p['ratio'] for p in points]
            distances = [p['distance'] for p in points]
            axes[0].scatter(ratios, distances, 
                             c=circuit_colors[circuit_type],
                             marker=circuit_markers[circuit_type],
                             s=80, alpha=0.7,
                             label=circuit_names[circuit_type])
    
    axes[0].set_xlabel('Max/Avg Lifetime Ratio (Pre-CAQR)', fontsize=12)
    axes[0].set_ylabel(f'Min Distance for {target_fidelity:.0%} Fidelity', fontsize=12)
    axes[0].set_title('Original Circuits: Universal Trend Across Algorithms', fontsize=14, fontweight='bold')
    axes[0].legend(loc='best', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Add trend line for original
    # if len(all_orig_points) > 3:
    #     all_ratios_orig = [p['ratio'] for p in all_orig_points]
    #     all_dists_orig = [p['distance'] for p in all_orig_points]
    #     z_orig = np.polyfit(all_ratios_orig, all_dists_orig, 2)
    #     p_orig = np.poly1d(z_orig)
    #     ratio_range = np.linspace(min(all_ratios_orig), max(all_ratios_orig), 100)
    #     axes[0, 0].plot(ratio_range, p_orig(ratio_range), 'k--', 
    #                    linewidth=2, alpha=0.5, label='Trend (2nd order)')
    #     axes[0, 0].legend(loc='best', fontsize=10)
    
    # Plot 2: CAQR circuits - all algorithms combined
    for circuit_type in ['bv', 'xor', 'ghz', 'simon', 'repetition']:
        points = [p for p in all_caqr_points if p['circuit'] == circuit_type]
        if points:
            ratios = [p['ratio'] for p in points]
            distances = [p['distance'] for p in points]
            axes[1].scatter(ratios, distances,
                             c=circuit_colors[circuit_type],
                             marker=circuit_markers[circuit_type],
                             s=80, alpha=0.7,
                             label=circuit_names[circuit_type])
    
    axes[1].set_xlabel('Max/Avg Lifetime Ratio (Post-CAQR)', fontsize=12)
    axes[1].set_ylabel(f'Min Distance for {target_fidelity:.0%} Fidelity', fontsize=12)
    axes[1].set_title('CAQR Circuits: Universal Trend Across Algorithms', fontsize=14, fontweight='bold')
    axes[1].legend(loc='best', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    # Add trend line for CAQR
    # if len(all_caqr_points) > 3:
    #     all_ratios_caqr = [p['ratio'] for p in all_caqr_points]
    #     all_dists_caqr = [p['distance'] for p in all_caqr_points]
    #     z_caqr = np.polyfit(all_ratios_caqr, all_dists_caqr, 2)
    #     p_caqr = np.poly1d(z_caqr)
    #     ratio_range_caqr = np.linspace(min(all_ratios_caqr), max(all_ratios_caqr), 100)
    #     axes[0, 1].plot(ratio_range_caqr, p_caqr(ratio_range_caqr), 'k--',
    #                    linewidth=2, alpha=0.5, label='Trend (2nd order)')
    #     axes[0, 1].legend(loc='best', fontsize=10)
    
    # Plot 3: Overlay - both original and CAQR
    # axes[1, 0].scatter([p['ratio'] for p in all_orig_points],
    #                   [p['distance'] for p in all_orig_points],
    #                   c='blue', marker='o', s=60, alpha=0.5, label='Original')
    # axes[1, 0].scatter([p['ratio'] for p in all_caqr_points],
    #                   [p['distance'] for p in all_caqr_points],
    #                   c='orange', marker='s', s=60, alpha=0.5, label='CAQR')
    
    # # Add trend lines
    # if len(all_orig_points) > 3:
    #     axes[1, 0].plot(ratio_range, p_orig(ratio_range), 'b--',
    #                    linewidth=2.5, alpha=0.7, label='Original Trend')
    # if len(all_caqr_points) > 3:
    #     axes[1, 0].plot(ratio_range_caqr, p_caqr(ratio_range_caqr), 'darkorange',
    #                    linestyle='--', linewidth=2.5, alpha=0.7, label='CAQR Trend')
    
    # axes[1, 0].set_xlabel('Max/Avg Lifetime Ratio', fontsize=12)
    # axes[1, 0].set_ylabel(f'Min Distance for {target_fidelity:.0%} Fidelity', fontsize=12)
    # axes[1, 0].set_title('Comparison: Original vs CAQR (All Algorithms)', fontsize=14, fontweight='bold')
    # axes[1, 0].legend(loc='best', fontsize=10)
    # axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Distance reduction histogram by circuit type
    # distance_reductions = defaultdict(list)
    # for circuit_type, results in all_results.items():
    #     for r in results:
    #         if r['min_d_orig'] <= max_distance and r['min_d_caqr'] <= max_distance:
    #             reduction = r['min_d_orig'] - r['min_d_caqr']
    #             distance_reductions[circuit_type].append(reduction)
    
    # x_pos = np.arange(len(distance_reductions))
    # avg_reductions = [np.mean(distance_reductions[ct]) if distance_reductions[ct] else 0 
    #                  for ct in ['bv', 'xor', 'ghz', 'simon', 'repetition']]
    # colors_bar = [circuit_colors[ct] for ct in ['bv', 'xor', 'ghz', 'simon', 'repetition']]
    
    # bars = axes[1, 1].bar(x_pos, avg_reductions, color=colors_bar, alpha=0.7, edgecolor='black')
    # axes[1, 1].set_xticks(x_pos)
    # axes[1, 1].set_xticklabels([circuit_names[ct] for ct in ['bv', 'xor', 'ghz', 'simon', 'repetition']])
    # axes[1, 1].set_ylabel('Average Distance Reduction', fontsize=12)
    # axes[1, 1].set_title('Average Distance Savings by Algorithm', fontsize=14, fontweight='bold')
    # axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    # axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    # for bar, val in zip(bars, avg_reductions):
    #     height = bar.get_height()
    #     axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
    #                    f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('combined_distance_vs_lifetime_all_algorithms.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n{'='*100}")
    print("Combined plot saved to: combined_distance_vs_lifetime_all_algorithms.png")
    print(f"{'='*100}")
    
    # Print summary statistics
    print("\n" + "="*100)
    print("SUMMARY STATISTICS: ALGORITHM-AGNOSTIC TRENDS")
    print("="*100)
    
    # print("\n--- ORIGINAL CIRCUITS ---")
    # orig_ratios = [p['ratio'] for p in all_orig_points]
    # orig_dists = [p['distance'] for p in all_orig_points]
    # if orig_ratios:
    #     print(f"Ratio range: {min(orig_ratios):.2f} - {max(orig_ratios):.2f}")
    #     print(f"Distance range: {min(orig_dists)} - {max(orig_dists)}")
    #     print(f"Avg ratio: {np.mean(orig_ratios):.2f}, Avg distance: {np.mean(orig_dists):.1f}")
    #     if len(orig_ratios) > 1:
    #         correlation = np.corrcoef(orig_ratios, orig_dists)[0, 1]
    #         print(f"Correlation (ratio vs distance): {correlation:.3f}")
    
    # print("\n--- CAQR CIRCUITS ---")
    # caqr_ratios = [p['ratio'] for p in all_caqr_points]
    # caqr_dists = [p['distance'] for p in all_caqr_points]
    # if caqr_ratios:
    #     print(f"Ratio range: {min(caqr_ratios):.2f} - {max(caqr_ratios):.2f}")
    #     print(f"Distance range: {min(caqr_dists)} - {max(caqr_dists)}")
    #     print(f"Avg ratio: {np.mean(caqr_ratios):.2f}, Avg distance: {np.mean(caqr_dists):.1f}")
    #     if len(caqr_ratios) > 1:
    #         correlation = np.corrcoef(caqr_ratios, caqr_dists)[0, 1]
    #         print(f"Correlation (ratio vs distance): {correlation:.3f}")
    
    # print("\n--- DISTANCE REDUCTIONS BY ALGORITHM ---")
    # for circuit_type in ['bv', 'xor', 'ghz', 'simon', 'repetition']:
    #     if distance_reductions[circuit_type]:
    #         avg_red = np.mean(distance_reductions[circuit_type])
    #         max_red = max(distance_reductions[circuit_type])
    #         print(f"{circuit_names[circuit_type]:12s}: avg={avg_red:5.2f}, max={max_red:2d}, "
    #               f"count={len(distance_reductions[circuit_type]):2d}")
    
    # Save all results
    import pickle
    with open('combined_all_algorithms_results.pkl', 'wb') as f:
        pickle.dump({
            'all_results': all_results,
            'orig_points': all_orig_points,
            'caqr_points': all_caqr_points,
            # 'distance_reductions': dict(distance_reductions)
        }, f)
    
    print("\nResults saved to: combined_all_algorithms_results.pkl")
    
    return all_results, all_orig_points, all_caqr_points


# Update main to include combined analysis
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python parallel.py <mode>")
        print("  mode: <circuit_type> [volume|distance] OR combined")
        print("  circuit_type: bv, xor, ghz, simon, repetition")
        print("  analysis_type: volume (default) or distance")
        print("  combined: run combined analysis across all algorithms")
        sys.exit(1)
    
    mode = sys.argv[1]
    
    if mode == "combined":
        # Run combined analysis across all circuit types
        print("\n" + "="*100)
        print("COMBINED ANALYSIS: ALL CIRCUIT TYPES")
        print("="*100)
        
        all_results, orig_points, caqr_points = analyze_combined_distance_vs_lifetime_ratio(
            target_fidelity=0.90,
            max_distance=25,
            rounds=10,
            p_phys_surface=1e-2,
            shots_pL=50_000,
            shots_algo=20_000,
        )
    else:
        # Original single-circuit analysis
        circuit_type = mode
        analysis_type = sys.argv[2] if len(sys.argv) > 2 else "volume"
        
        if analysis_type == "distance":
            lengths = [n for n in range(3, 15, 2)]
            
            print(f"\n{'='*90}")
            print(f"MIN DISTANCE vs LIFETIME RATIO - {circuit_type.upper()}")
            print(f"{'='*90}")
            
            results = analyze_min_distance_vs_lifetime_ratio(
                string_lengths=lengths,
                target_fidelity=0.90,
                max_distance=21,
                rounds=10,
                p_phys_surface=1e-2,
                shots_pL=50_000,
                shots_algo=20_000,
                circuit_type=circuit_type,
            )
        else:
            distances = [n for n in range(3, 17, 2)]
            lengths = [n for n in range(3, 15, 2)]
            
            print(f"\n{'='*90}")
            print(f"FIDELITY vs SPACE-TIME VOLUME - {circuit_type.upper()}")
            print(f"{'='*90}")
            
            results = analyze_fidelity_vs_space_time(
                string_lengths=lengths,
                distances=distances,
                rounds=10,
                p_phys_surface=1e-2,
                shots_pL=50_000,
                shots_algo=20_000,
                circuit_type=circuit_type,
            )
        
        import pickle
        with open(f'{circuit_type}_{analysis_type}_results.pkl', 'wb') as f:
            pickle.dump(results, f)
        
        print(f"\nResults saved to {circuit_type}_{analysis_type}_results.pkl")