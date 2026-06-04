import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from qiskit.circuit import Parameter, ParameterVector, QuantumCircuit
from qiskit.circuit.library import z_feature_map, unitary_overlap, zz_feature_map
from qiskit.primitives import StatevectorSampler
from qiskit_ibm_runtime import Options, Session, Sampler
from sklearn.svm import SVC

#---------------------------------------------------------------------------------Single Feature Example--------------------------------------------------------------------

# Two mock data points, including category labels, as in training
small_data = [
    [-0.194, 0.114, -0.006, 0.301, -0.359, -0.088, -0.156, 0.342, -0.016, 0.143, 1],
    [-0.1, 0.002, 0.244, 0.127, -0.064, -0.086, 0.072, 0.043, -0.053, 0.02, -1],
]

# Data points with labels removed for inner product
training_data = [small_data[0][:-1], small_data[1][:-1]]

f_map = z_feature_map(feature_dimension=np.shape(training_data)[1])

unitary_1 = f_map.assign_parameters(training_data[0])
unitary_2 = f_map.assign_parameters(training_data[1])

unitary_overlap_circ = unitary_overlap(unitary_1, unitary_2)
unitary_overlap_circ.measure_all()

unitary_overlap_circ.draw(output = 'mpl', filename = 'Quantum Kernels/UnitaryOverlapCircuit')

# Run this for a simulator

# Evaluate the problem using state vector-based primitives from Qiskit
sampler = StatevectorSampler()
results = sampler.run([unitary_overlap_circ], shots=10000).result()
# counts associated with a state labeled by bit results such as |001101...01>.
counts_bit = results[0].data.meas.get_counts()
# the same counts but labeled by integer equivalent of the above bit string
counts = results[0].data.meas.get_int_counts()

print(counts_bit)
print(counts)
print(counts.get(0, 0.0) / 10000)


#---------------------------------------------------------------------------------Full Kernel Matrix--------------------------------------------------------------------



# Lines of n=14 feature vectors with the final element of each line being the binary 
# classification marker of \pm 1 used for labelling.

# The first 120 lines of data are used and is split into 75% training data and 25% testing data for validation

df = pd.read_csv("Quantum Kernels/dataset_graph7.csv", sep=",", header=None)

training_data_size = 90
testing_data_size = 30

training_data = df.values[0:training_data_size, :-1]
training_labels = df.values[0:training_data_size, -1]

testing_data = df.values[training_data_size:training_data_size+testing_data_size, :-1]
testing_labels = df.values[training_data_size:training_data_size+testing_data_size, -1]

# Initialise and empty kernel matrix of the appropriate size for the training and testing data

num_samples = np.shape(training_data)[0]

kernel_matrix = np.full((num_samples, num_samples), np.nan)
test_matrix = np.full((testing_data_size, num_samples), np.nan)

# Prepare the feature map for computing the overlap
num_features = np.shape(training_data)[1]
num_qubits = int(num_features/2)

# Custom feature map provided
entangler_map = [[0,2],[3,4],[2,5],[1,4],[2,3],[4,6]]

f_map = QuantumCircuit(num_qubits)
training_param = Parameter("T")
feature_params = ParameterVector("x", 2*num_qubits)
f_map.ry(training_param, f_map.qubits)
for cz in entangler_map:
    f_map.cz(cz[0],cz[1])
for i in range(num_qubits):
    f_map.rz(-2*feature_params[2*i+1],i)
    f_map.rx(-2*feature_params[2*i],i)

f_map.draw(output = 'mpl', filename = 'Quantum Kernels/CustomFeatureMap')

# Evaluate using state vector-based qiskit primatives

run = False #on/off flag to save computation time when writing later sections

if run:

    sampler = StatevectorSampler()

    for i in range(training_data_size):
        for j in range(i+1, training_data_size):
            
            # generate the feature map circuits for feature i and j
            unitary1 = f_map.assign_parameters([np.pi/2]+list(training_data[i])) #Parameters assigned T = np.pi/2, x = list(training_data[i])
            unitary2 = f_map.assign_parameters([np.pi/2]+list(training_data[j]))

            # create the overlap circuit for feature i and j
            overlap_circuit = unitary_overlap(unitary1, unitary2)
            overlap_circuit.measure_all()

            counts = (sampler.run([overlap_circuit], shots = 10000).result()[0].data.meas.get_int_counts())

            # assign the elements of the kernel matrix to by the probability of the zero state
            # i.e. an overlap metric for the features
            kernel_matrix[i,j] = counts.get(0,0.0)/10000
            kernel_matrix[j,i] = counts.get(0,0.0)/10000
        
        kernel_matrix[i,i] = 1 #diagonal elements with an overlap of 1

    print("Kernel Matrix Computed, Training Completed")

    # Similar process to fill the test matrix

    for i in range(testing_data_size):
        for j in range(training_data_size):
            unitary1 = f_map.assign_parameters([np.pi/2]+list(testing_data[i]))
            unitary2 = f_map.assign_parameters([np.pi/2]+list(training_data[j]))

            overlap_circuit = unitary_overlap(unitary1, unitary2)
            overlap_circuit.measure_all()

            counts = (sampler.run([overlap_circuit], shots = 10000).result()[0].data.meas.get_int_counts())

            test_matrix[i,j] = counts.get(0,0.0)/10000

    print("Test Matrix Computed")
    print(kernel_matrix)
    print(test_matrix)

    # Now apply a classical machine learning algorithm to make predictions on the 
    # test matrix using the precomputed kernel matrix

    qml_svc = SVC(kernel = 'precomputed') #support vector classifier

    # give the precmputed kernel matrix and labels to the SVC
    qml_svc.fit(kernel_matrix, training_labels)

    # use .score to test the kernel data on the testing data
    qml_score = qml_svc.score(test_matrix, testing_labels)

    print("Precomputed kernel classification score: "+str(qml_score)) #1.0 = Perfectly classified :)


#---------------------------------------------------------------------------------Larger Single Feature--------------------------------------------------------------------

# Two mock data points, including category labels, as in training

large_data = [
    [
        -0.028,
        -1.49,
        -1.698,
        0.107,
        -1.536,
        -1.538,
        -1.356,
        -1.514,
        -0.109,
        -1.8,
        -0.122,
        -1.651,
        -1.955,
        -0.123,
        -1.732,
        0.091,
        -0.048,
        -0.128,
        -0.026,
        0.082,
        -1.263,
        0.065,
        0.004,
        -0.055,
        -0.08,
        -0.173,
        -1.734,
        -0.39,
        -1.451,
        0.078,
        -1.578,
        -0.025,
        -0.184,
        -0.119,
        -1.336,
        0.055,
        -0.204,
        -1.578,
        0.132,
        -0.121,
        -1.599,
        -0.187,
        -1,
    ],
    [
        -1.414,
        -1.439,
        -1.606,
        0.246,
        -1.673,
        0.002,
        -1.317,
        -1.262,
        -0.178,
        -1.814,
        0.013,
        -1.619,
        -1.86,
        -0.25,
        -0.212,
        -0.214,
        -0.033,
        0.071,
        -0.11,
        -1.607,
        0.441,
        -0.143,
        -0.009,
        -1.655,
        -1.579,
        0.381,
        -1.86,
        -0.079,
        -0.088,
        -0.058,
        -1.481,
        -0.064,
        -0.065,
        -1.507,
        0.177,
        -0.131,
        -0.153,
        0.07,
        -1.627,
        0.593,
        -1.547,
        -0.16,
        -1,
    ],
]

training_data = [large_data[0][:-1], large_data[1][:-1]]

# Try the zz_feature map to test its circuit depth comparative to feature size using larger features

f_map = zz_feature_map(feature_dimension=np.shape(training_data)[1], entanglement='linear', reps = 1)

unitary1 = f_map.assign_parameters(training_data[0])
unitary2 = f_map.assign_parameters(training_data[1])

overlap_circuit = unitary_overlap(unitary1,unitary2)
overlap_circuit.measure_all()

overlap_circuit.draw(output = 'mpl', filename = 'Quantum Kernels/LargeFeatureOverlapCircuit')
depth = overlap_circuit.decompose(reps =2).depth()
two_qubit_depth = overlap_circuit.decompose().depth(lambda instr: len(instr.qubits) > 1)

print(depth) #251
print(two_qubit_depth) #165

# two qubit depth > 100 before any kind of transpilation is not going to happen on modern hardware in reasonabel time with reasonable error
# hence the importance of custom feature maps

# Custom feature map:

entangler_map = [
    [3, 4],
    [2, 5],
    [1, 4],
    [2, 3],
    [4, 6],
    [7, 9],
    [10, 11],
    [9, 12],
    [8, 11],
    [9, 10],
    [11, 13],
    [14, 16],
    [17, 18],
    [16, 19],
    [15, 18],
    [16, 17],
    [18, 20],
]

num_features = np.shape(training_data)[1]
num_qubits = int(num_features/2)

f_map = QuantumCircuit(num_qubits)
training_parameter = Parameter("T")
feature_parameters = ParameterVector("x", 2*num_qubits)
f_map.ry(training_parameter, f_map.qubits)
for cz in entangler_map:
    f_map.cz(cz[0], cz[1])
for i in range(num_qubits):
    f_map.rz(-2*feature_parameters[2*i+1], i)
    f_map.rx(-2*feature_parameters[2*i], i)


# generate the feature circuits and their overlap circuit

unitary1 = f_map.assign_parameters([np.pi/2]+list(training_data[0]))
unitary2 = f_map.assign_parameters([np.pi/2]+list(training_data[1]))

overlap_circuit = unitary_overlap(unitary1,unitary2)
overlap_circuit.measure_all()

overlap_circuit.draw(output = 'mpl', filename = 'Quantum Kernels/LargeFeaturesCustomMap')

# n=42 features => num_qubits = 21 which is too large to run on the StatevectorSampler simulator. From the provided results using ibm_brisbane and heavy optimisation
# the overlap metric for this data is ~0.01.

# sampler = StatevectorSampler()

# counts = (sampler.run([overlap_circuit], shots = 10000).result()[0].data.meas.get_int_counts())

# print(counts.get(0,0.0)/10000) #~0.01



