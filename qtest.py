import numpy as np
import qiskit as qk
from scipy.optimize import minimize

# n_qubits = 1
# n_cbits = 1
# qreg = qk.QuantumRegister(n_qubits)
# creg = qk.ClassicalRegister(n_cbits)
# circuit = qk.QuantumCircuit(qreg, creg)

# circuit.draw()

# circuit.x(qreg[0])
# circuit.draw("mpl")

# print(circuit)

qc = qk.QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.draw("mpl")
print(qc)