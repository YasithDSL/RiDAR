**Authors:**  
Yasith Medagama Disanayakage & Deniz Alkan  
*University of Cambridge*

---

## Project Overview

This project investigates the efficacy of **Compiler-Assisted Qubit Reuse (CaQR)** within the context of **Fault-Tolerant Quantum Computing (FTQC)**. 
The framework benchmarks various quantum algorithms by simulating their execution on a surface code substrate using **Stim** for efficient Clifford circuit simulation and **Qiskit** for circuit construction and manipulation.

CaQR GitHub: https://github.com/ruadapt/CaQR

### Key Metrics
- **Physical Space-Time Volume ($V_{ST}$):** The product of active physical qubits and circuit duration (in code cycles).
- **Logical Fidelity:** The success probability of the algorithm after error correction.
- **Lifetime Ratio:** The ratio of maximum qubit lifetime to average qubit lifetime, used to predict decoherence impacts.

## Supported Algorithms

The benchmarking suite consists of the following quantum algorithms/circuits:
- **Bernstein-Vazirani (BV)**
- **XOR_n** (Parity calculation)
- **GHZ State Preparation**
- **Simon's Algorithm**
- **Repetition Code**

## Installation

Ensure you have the required Python packages installed:

```bash
pip install -r requirements.txt
```

## Usage

The main entry point for the analysis is `parallel.py`. You can run benchmarks for individual algorithms or perform a combined analysis across all implemented circuits.

### Running a Combined Analysis
To run the comprehensive analysis across all algorithms and generate combined plots (e.g., Distance vs. Lifetime Ratio):

```bash
python parallel.py combined
```

### Running Single Algorithm Benchmarks
To analyze the Fidelity vs. Space-Time Volume for a specific algorithm:

```bash
python parallel.py <circuit_type>
```

**Available `<circuit_type>` options:**
- `bv`
- `xor`
- `ghz`
- `simon`
- `repetition`

**Example:**
```bash
python parallel.py bv
```

## Project Structure

- **`parallel.py`**: Main execution script for running benchmarks and generating plots.
- **`clean_notebook.ipynb`**: Jupyter notebook for interactive analysis and prototyping.
- **`circuit_analysis.py`**: Core logic for CAQR implementation, including finding reuse pairs and modifying circuits.
- **`quantum_utils.py`**: Utility functions for quantum circuit operations.

Both circuit_analysis.py and quantum_utils.py are from or adapted from https://github.com/ruadapt/CaQR


