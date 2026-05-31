# Qiskit built in feature map examples

from qiskit import QuantumCircuit
from qiskit.circuit.library import efficient_su2, z_feature_map, zz_feature_map
import numpy as np

# Efficient SU2 can map many features to few qubits

eff_su2 = efficient_su2(num_qubits=2, reps = 1, insert_barriers=True)
eff_su2.decompose().draw(output = 'mpl', filename = 'Data Encoding/EfficientSU2')

# The resulting state of this circuit is in terms of the 4 computational basis states
# but can encode 8 features in the real and imaginary components of the 4 amplitudes.

# Example encoding the feature vector x in a 3-qubit Efficient SU2 circuit

x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]

eff_su2_x = efficient_su2(num_qubits=3, reps = 1, insert_barriers=True)
eff_su2_x.assign_parameters(x)
eff_su2_x.decompose().draw(output = 'mpl', filename = 'Data Encoding/EfficientSU2FeatureExample')


# Z feature map example

x = [np.pi/2, np.pi/3]

# single repetition encoding the feature x
z_map = QuantumCircuit(2)
for i,f in enumerate(x):
    z_map.h(i)
    z_map.rz(f,i)

z_map.draw(output = 'mpl', filename = 'Data Encoding/ZMapExampleManual')

# 3 repetitions encoding the feature x using the z_feature circuit
# Qiskit z_feature_map requires parameters to be passed with an additional factor of 0.5
# as it applies phase gates for parameter A as P(2A)
x_ = [i/2 for i in x]

z_map_qiskit = z_feature_map(feature_dimension=2, reps = 3)
z_map_qiskit.assign_parameters(x_)
z_map_qiskit.draw(output = 'mpl', filename = 'Data Encoding/ZMapExampleQiskit')

# ZZ-feature map

x = np.pi

ZZ_manual = QuantumCircuit(2)
ZZ_manual.rzz(x,0,1)
ZZ_manual.draw(output = 'mpl', filename = 'Data Encoding/ZZMapExampleManual')


# ZZ-feature map qiskit

# feature vector of dimension 2

x = [np.pi, np.pi/2]

ZZ_qiskit = zz_feature_map(feature_dimension=2, reps = 1)
ZZ_qiskit.assign_parameters(x)
ZZ_qiskit.draw(output = 'mpl', filename = 'Data Encoding/ZZMapExampleQiskit')

# feature vector of dimension 4

x = [np.pi, np.pi/2, np.pi, np.pi/3]

ZZ_qiskit = zz_feature_map(feature_dimension=4, reps = 1)
ZZ_qiskit.assign_parameters(x)
ZZ_qiskit.draw(output = 'mpl', filename = 'Data Encoding/ZZMapExampleQiskit2')

# Pauli feature map

from qiskit.circuit.library import pauli_feature_map

x = [np.pi/2, np.pi/3, np.pi/4]

pauli_map = pauli_feature_map(feature_dimension=3, entanglement="linear", reps=1, paulis=["Y", "XX"])
pauli_map.assign_parameters(x)
pauli_map.decompose().draw(output = 'mpl', filename = 'Data Encoding/PauliMapExample')



