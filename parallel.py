from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGOpNode
from circuit_analysis import find_qubit_reuse_pairs, modify_circuit, last_index_operation, first_index_operation
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import stim
import sys
import pymatching
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
import numpy as np

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

def qiskit_to_stim_circuit_with_noise(
    qc: QuantumCircuit,
    p1: float = 0.0,
    p2: float = 0.0,
    p_measure: float = 0.01,
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

def run_bv_on_stim(
    qc: QuantumCircuit,
    secret_string: str,
    p1: float = 0.0,
    p2: float = 0.0,
    shots: int = 20_000,
) -> float:   
    n = len(secret_string)

    stim_circuit = qiskit_to_stim_circuit_with_noise(qc, p1=p1, p2=p2)

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

    sampler = stim_circuit.compile_sampler()
    samples = sampler.sample(shots)

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

def get_active_qubit_count(qc: QuantumCircuit) -> int:
    used_qubits = set()
    for instruction in qc.data:
        for q in instruction.qubits:
            used_qubits.add(q)
    return len(used_qubits)

def physical_space_time_volume(qc, distance: int, rounds_per_layer: int = None) -> int:
    n_logical = get_active_qubit_count(qc) # otherwise reuse is not credited
    
    if rounds_per_layer is None:
        rounds_per_layer = distance
    
    space = n_logical * (distance ** 2)
    time = get_logical_circuit_time(qc)
    
    return space * time

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

def build_simon_circuit(n: int, s: str) -> QuantumCircuit:
    if len(s) != n:
        raise ValueError(f"Secret string length {len(s)} must match n={n}")
    
    qr_in = QuantumRegister(n, 'x')
    qr_out = QuantumRegister(n, 'y')
    cr = ClassicalRegister(n, 'meas')
    
    qc = QuantumCircuit(qr_in, qr_out, cr)
    
    qc.h(qr_in)
    
    s_rev = s[::-1]
    
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
    
    qc.h(qr_in)
    
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
    
    sampler = stim_circuit.compile_sampler()
    samples = sampler.sample(shots)
    
    s_int = int(secret_s, 2)
    valid_count = 0
    
    for shot in samples:
        cvals = [0] * n
        for meas_idx, c_idx in meas_map:
            if c_idx < n:
                cvals[c_idx] = int(shot[meas_idx])
        
        z_int = int(''.join(str(cvals[i]) for i in range(n-1, -1, -1)), 2)
        
        dot_product = bin(z_int & s_int).count('1') % 2
        
        if dot_product == 0:
            valid_count += 1
    
    fidelity = valid_count / shots
    return fidelity




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

    if logical_state == "1":
        qc.x(0)

    for j in range(1, distance):
        qc.cx(0, j)

    if inject_error_qubit is not None:
        qc.x(inject_error_qubit)

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
    
    sampler = stim_circuit.compile_sampler()
    samples = sampler.sample(shots)
    
    correct_count = 0
    expected_bit = int(logical_state)
    
    for shot in samples:
        cvals = [0] * distance
        for meas_idx, c_idx in meas_map:
            if c_idx < distance:
                cvals[c_idx] = int(shot[meas_idx])
        
        ones_count = sum(cvals)
        decoded_bit = 1 if ones_count > distance // 2 else 0
        
        if decoded_bit == expected_bit:
            correct_count += 1
    
    fidelity = correct_count / shots
    return fidelity

def get_idle_times(qc: QuantumCircuit):
    dag = circuit_to_dag(qc)
    all_lifetimes = [] #all idle times
    dynamic_lifetimes = [0] * qc.num_qubits #current idle times
    measured_qubits = []

    for depth, layer in enumerate(dag.layers()):
        longest_time_op = 1
        used_qubits = [] #indices of qubits that undergo operations during the layer
        for node in layer["graph"].op_nodes():
            #assume that a barrier cannot be at the same time as any other operation
            if node.name == 'barrier':
                longest_time_op = 0
                break
            elif node.name in two_beat_gates:
                longest_time_op = max(2, longest_time_op)
            elif node.name in three_beat_gates:
                longest_time_op = max(3, longest_time_op)

            for q in node.qargs:
                index = qc.qubits.index(q)
                if index not in used_qubits:
                    used_qubits.append(index)
        
        for node in layer["graph"].op_nodes():
            if node.name == 'barrier':
                longest_time_op = 0
                break

            if node.name == 'measure':
                for q in node.qargs:
                    index = qc.qubits.index(q)
                    all_lifetimes.append(dynamic_lifetimes[index])
                    dynamic_lifetimes[index] = 0
                    #fixes the qubit lifetime so it will not be incremented later, unless there is a reset
                    measured_qubits.append(index)
            elif node.name == 'reset':
                for q in node.qargs:
                    index = qc.qubits.index(q)
                    if index in measured_qubits:
                        measured_qubits.remove(index)
            
            # we should still do this for a reset gate:
            if node.name != 'measure':
                node_time = 3 if node.name in three_beat_gates else 2 if node.name in two_beat_gates else 1
                for q in node.qargs:
                    index = qc.qubits.index(q)
                    dynamic_lifetimes[index] += longest_time_op - node_time
        
        # add idling noise to all qubits not operated on in the layer
        for i in range(qc.num_qubits):
            if i not in used_qubits and i not in measured_qubits:
                dynamic_lifetimes[i] += longest_time_op
        
    all_lifetimes += list(filter(lambda x: x != 0, dynamic_lifetimes))
    return {'idle_times': all_lifetimes}

def get_decoherence_fidelities(lifetimes, distance: int):
    fidelities = []
    T1 = 20000 #T1 time in SC is ~200µs and single qubit gate is ~10ns
    #https://www.spinquanta.com/news-detail/ultimate-guide-to-coherence-time
    #https://postquantum.com/quantum-modalities/superconducting-qubits
    max_e_phys = 0.1 #derived from formula in Camps, Rrapaj, Klymko et al.
    
    for lifetime in lifetimes:
        #formula from Camps, Rrapaj, Klymko et al. 
        #Quantum Computing Technology Roadmaps and Capability Assessment for Scientific
        #Computing - An analysis of use cases from the NERSC workload
        e_phys = min(max_e_phys, 1-np.exp(-lifetime/T1))
        f_log = 1 - 0.1 * ((100*(e_phys))**((distance+1)/2))
        #maximum logical error should not surpass maximum physical error
        fidelities.append(max(f_log, 1-max_e_phys))

    return fidelities

def get_total_decoherence_fidelity(fidelities: list):
    total_fidelity = 1
    for f in fidelities:
        total_fidelity *= f
    return total_fidelity

#one_beat is assumed to be all other gates
two_beat_gates = ['cx'] #we assume that the circuits are small enough that cnots do not require too much overhead
three_beat_gates = ['h']

def get_logical_circuit_time(qc: QuantumCircuit):
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

        # Build logical circuit based on type
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
            qc_seq = build_repetition_code_circuit(
                distance=n,
                logical_state=logical_state,
                inject_error_qubit=None
            )
            expected_outcome = logical_state
            code_distance_for_rep = n
            
        else:
            raise ValueError(f"Unknown circuit_type: {circuit_type}")

        # Apply CaQR
        qc_caqr, iter_count, _ = apply_caqr(qc_seq)

        # Scan code distances
        for d in tqdm(distances, desc=f"Distances (n={n})", leave=False):

            pL = estimate_pL_surface_code(
                distance=d,
                rounds=rounds,
                p_phys=p_phys_surface,
                shots=shots_pL,
            )

            p1 = pL
            p2 = 2 * pL

            # Compute fidelity for original circuit
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
                
            else:
                F_orig = run_bv_on_stim(qc_seq, expected_outcome, p1=p1, p2=p2,
                                       shots=shots_algo)

            # Compute fidelity for CaQR circuit
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
                
            else:
                F_caqr = run_bv_on_stim(qc_caqr, expected_outcome, p1=p1, p2=p2,
                                       shots=shots_algo)

            # Compute approximate logical space–time volume
            V_orig = physical_space_time_volume(qc_seq, distance=d)
            V_caqr = physical_space_time_volume(qc_caqr, distance=d)

            lifetimes_pre_caqr = get_idle_times(qc_seq)['idle_times']
            lifetimes_post_caqr = get_idle_times(qc_caqr)['idle_times']
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

    fig, ax = plt.subplots(figsize=(8, 6))

    orig_by_d = defaultdict(lambda: {'V': [], 'F': []})
    caqr_by_d = defaultdict(lambda: {'V': [], 'F': []})
    
    for r in results:
        d = r['d']
        orig_by_d[d]['V'].append(r['V_orig'])
        orig_by_d[d]['F'].append(r['F_orig'])
        caqr_by_d[d]['V'].append(r['V_caqr'])
        caqr_by_d[d]['F'].append(r['F_caqr'])
    
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
    ax.set_yscale("log")

    ax.plot(V_o_avg, F_o_avg, 'o-', color='blue', linewidth=2, markersize=8, 
            label=f"Original (avg n={string_lengths[0]}...{string_lengths[-1]})", alpha=0.8)
    ax.plot(V_c_avg, F_c_avg, 's-', color='orange', linewidth=2, markersize=8, 
            label=f"CaQR (avg n={string_lengths[0]}...{string_lengths[-1]})", alpha=0.8)
    
    for i, d in enumerate(distances_used):
        ax.annotate(f'd={d}', (V_o_avg[i], F_o_avg[i]), 
                   fontsize=8, alpha=0.6, xytext=(5, 5), 
                   textcoords='offset points')
        ax.annotate(f'd={d}', (V_c_avg[i], F_c_avg[i]), 
                   fontsize=8, alpha=0.6, xytext=(5, -10), 
                   textcoords='offset points')

    ax.set_xlabel("Physical Space–Time Volume (d² × beats)", fontsize=12)
    ax.set_ylabel("Circuit Fidelity (%)", fontsize=12)
    ax.set_title(f"{circuit_name}: Circuit Fidelity vs. Space–Time Volume (Averages)", fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(f'{circuit_type}_fidelity_vs_volume.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to {circuit_type}_fidelity_vs_volume.png")
    
    return results


def get_average_lifetimes(qc: QuantumCircuit) -> dict:
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
    max_lt = np.max(all_lifetimes)
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
    for d in range(3, max_distance + 1, 2):
        pL = estimate_pL_surface_code(
            distance=d,
            rounds=rounds,
            p_phys=p_phys_surface,
            shots=shots_pL,
        )
        p1 = pL
        p2 = 2 * pL
        
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
            get_decoherence_fidelities(get_idle_times(qc)['idle_times'], distance=d)
        )
        print(decoherence)
        fidelity *= decoherence

        if fidelity >= target_fidelity:
            return d
    
    return max_distance + 2  # Could not achieve target

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
            
            # Build circuit
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
            
            # Apply CaQR
            qc_caqr, iter_count, _ = apply_caqr(qc_seq)
            
            # Compute lifetime metrics
            lifetimes_orig = get_average_lifetimes(qc_seq)
            lifetimes_caqr = get_average_lifetimes(qc_caqr)
            
            # Find minimum distances
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
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
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
    
    axes[0].set_xlabel('Max/Avg Lifetime Ratio (Pre-CaQR)', fontsize=12)
    axes[0].set_ylabel(f'Min Distance for {target_fidelity:.0%} Circuit Fidelity', fontsize=12)
    axes[0].set_title('Original Circuits: Surface Code Distance vs. Lifetime Ratios', fontsize=14, fontweight='bold')
    axes[0].legend(loc='best', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
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
    
    axes[1].set_xlabel('Max/Avg Lifetime Ratio (Post-CaQR)', fontsize=12)
    axes[1].set_ylabel(f'Min Distance for {target_fidelity:.0%} Circuit Fidelity', fontsize=12)
    axes[1].set_title('CaQR Circuits: Surface Code Distance vs. Lifetime Ratios', fontsize=14, fontweight='bold')
    axes[1].legend(loc='best', fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('combined_distance_vs_lifetime_all_algorithms.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n{'='*100}")
    print("Combined plot saved to: combined_distance_vs_lifetime_all_algorithms.png")
    print(f"{'='*100}")
    
    print("\n" + "="*100)
    print("SUMMARY STATISTICS: ALGORITHM-AGNOSTIC TRENDS")
    print("="*100)
    
    return all_results, all_orig_points, all_caqr_points


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python parallel.py <mode>")
        print("  mode: [<circuit_type> volume] OR combined")
        print("  circuit_type: bv, xor, ghz, simon, repetition")
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