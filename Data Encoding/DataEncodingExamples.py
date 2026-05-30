# Data encoding

from qiskit import QuantumCircuit
import numpy as np
from qiskit.quantum_info import Statevector
from qiskit.visualization.bloch import Bloch
from qiskit.visualization.state_visualization import _bloch_multivector_data

# Example binary encoding of integer data points

X = 5
Y = 7

x_bits = [int(b) for b in format(X,'04b')]
y_bits = [int(b) for b in format(Y,'04b')]

bits = x_bits + y_bits

binary_circuit = QuantumCircuit(len(bits))

for ind, bit in enumerate(bits):
    if bit == 1:
        binary_circuit.x(ind)

binary_circuit.draw(output = 'mpl', filename = 'Data Encoding/BinaryEncodingExample')

# Example amplitude encoding of feature vector

f = [5,7,2]

# Extend to have 2^n elements the closest positive integer n (n=2) withouf affecting normalisation

f.append(0)

# Normalisation

alpha = np.linalg.norm(f)

desired_state = f/alpha

amplitude_circuit = QuantumCircuit(2) # num_qubits > log_2(len(f)) is the requirement. 2 > log_2(3).
amplitude_circuit.initialize(desired_state)

amplitude_circuit.draw(output = 'mpl', filename = 'Data Encoding/AmplitudeEncodingExample')

# General amplitude encoding method

def amplitude_encoding(feature_vector):
    # create a string of the feature vector for unique file naming when saving the circuit
    vector_name = str(feature_vector)

    # Extend the list to have 2^n elements for the closest positive integer n
    if np.log2(len(feature_vector)) - int(np.log2(len(feature_vector))) < 0.5:
        new_len = 2**(np.round(np.log2(len(feature_vector))+0.5)) # ensure it rounds up
    else:
        new_len = 2**np.round(np.log2(len(feature_vector)))
    while len(feature_vector) < new_len:
        feature_vector.append(0)

    # normalisation
    norm = np.linalg.norm(feature_vector)

    feature_vector = feature_vector/norm

    # generate the encoding circuit
    encoded_circuit = QuantumCircuit(np.log2(new_len))
    encoded_circuit.initialize(feature_vector)

    encoded_circuit.draw(output = 'mpl', filename = 'Data Encoding/AmplitudeEncoding'+vector_name)

    return encoded_circuit

# Example angle encoding of data point x = pi/2


# data shoudl be normalised into the region [0,2pi) and therefore to achieve encoding 
# without the loss of information 

x = np.pi/2

qc = QuantumCircuit(1)
initial_state = Statevector.from_instruction(qc)
qc.ry(x,0)
prepared_state = Statevector.from_instruction(qc)
states = [initial_state, prepared_state]


def plot_Nstates(states, axis = None, plot_trace_points=True, filename = 'BlochSphere'):
    """This function plots N states to 1 Bloch sphere"""
    bloch_vecs = [_bloch_multivector_data(s)[0] for s in states]

    if axis is None:
        bloch_plot = Bloch()
    else:
        bloch_plot = Bloch(axes=axis)

    bloch_plot.add_vectors(bloch_vecs)

    if len(states) > 1:

        def rgba_map(x, num):
            g = (0.95 - 0.05) / (num - 1)
            i = 0.95 - g * num
            y = g * x + i
            return (0.0, y, 0.0, 0.7)

        num = len(states)
        bloch_plot.vector_color = [rgba_map(x, num) for x in range(1, num + 1)]

    bloch_plot.vector_width = 3
    bloch_plot.vector_style = "simple"

    if plot_trace_points:

        def trace_points(bloch_vec1, bloch_vec2):
            # bloch_vec = (x,y,z)
            n_points = 15
            thetas = np.arccos([bloch_vec1[2], bloch_vec2[2]])
            phis = np.arctan2(
                [bloch_vec1[1], bloch_vec2[1]], [bloch_vec1[0], bloch_vec2[0]]
            )
            if phis[1] < 0:
                phis[1] = phis[1] + 2 * pi
            angles0 = np.linspace(phis[0], phis[1], n_points)
            angles1 = np.linspace(thetas[0], thetas[1], n_points)

            xp = np.cos(angles0) * np.sin(angles1)
            yp = np.sin(angles0) * np.sin(angles1)
            zp = np.cos(angles1)
            pnts = [xp, yp, zp]
            bloch_plot.add_points(pnts)
            bloch_plot.point_color = "k"
            bloch_plot.point_size = [4] * len(bloch_plot.points)
            bloch_plot.point_marker = ["o"]

        for i in range(len(bloch_vecs) - 1):
            trace_points(bloch_vecs[i], bloch_vecs[i + 1])

    bloch_plot.sphere_alpha = 0.05
    bloch_plot.frame_alpha = 0.15
    bloch_plot.figsize = [4, 4]

    bloch_plot.save(filename)

plot_Nstates(states, filename = 'Data Encoding/AngleEncodingExample')

# Encoding of the feature (0, pi/4, pi/2) in angle encoding

feat = [0, np.pi/4, np.pi/2]

qc = QuantumCircuit(len(feat))
for ind, f in enumerate(feat):
    qc.ry(f,ind)

qc.draw(output = 'mpl', filename = 'Data Encoding/FeatureAngleEncoding')


# Phase encoding of data point pi/2

# Initialise qubits in |+> state and apply rotation around the z-axis for data normalised in [0,2pi)

x = np.pi/2

qc = QuantumCircuit(1)
qc.h(0)
intial_state = Statevector.from_instruction(qc)
qc.rz(x,0)
prepared_state = Statevector.from_instruction(qc)

states = intial_state, prepared_state

plot_Nstates(states, filename = 'Data Encoding/PhaseEncodingExample')

# Phase encoding of the feature f = (4,8,5,9,8,6,2,9,2,5,7,0)

f = [4,8,5,9,8,6,2,9,2,5,7,0]

qc = QuantumCircuit(len(f))
qc.h([i for i in range(len(f))])
for ind, x in enumerate(f):
    qc.rz(x,ind)
qc.draw(output='mpl', filename = 'Data Encoding/FeaturePhaseEncoding')

# Dense Angle Encoding (DAE)

# Combines angle encoding and phase encoding to encode 2 data points on one qubit
# |x,y> = cos(x)|0> + e^(iy)sin(x)|1>

# Example encoding the two data points x_1 = 3pi/8 and x_2 = pi/2 on a singular qubit

x_1 = 3*np.pi/8
x_2 = np.pi/2

qc = QuantumCircuit(1)
state1 = Statevector.from_instruction(qc)
qc.ry(x_1,0)
state2 = Statevector.from_instruction(qc)
qc.rz(x_2,0)
state3 = Statevector.from_instruction(qc)

states = state1, state2, state3

plot_Nstates(states, filename = 'Data Encoding/DAEExample')

# Encoding the feature x = (4,8,5,9,8,6,2,9,2,5,7,0,3,7,5) in DAE

x = [4,8,5,9,8,6,2,9,2,5,7,0,3,7,5]
if len(x)%2 != 0:
    x.append(0)

# normalise the feature into [0,2pi]
x_normalised = [2*np.pi*element/max(x) for element in x]

n = int(len(x)/2)

qc = QuantumCircuit(n)
for i in range(n):
    qc.ry(x_normalised[i],i)
    qc.rz(x_normalised[i+n],i)

qc.draw(output = 'mpl', filename = 'Data Encoding/FeatureDAE')
    



