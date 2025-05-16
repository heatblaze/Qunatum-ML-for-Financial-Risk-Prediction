from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
import numpy as np
from sklearn.svm import SVC

def build_quantum_reservoir(n_qubits, data_point):
    qc = QuantumCircuit(n_qubits)
    for i, x in enumerate(data_point):
        qc.rx(x, i % n_qubits)
    qc.h(range(n_qubits))
    qc.barrier()
    return qc

def extract_features(qc, n_shots=256):
    simulator = Aer.get_backend('qasm_simulator')
    qc.measure_all()
    transpiled = transpile(qc, simulator)
    result = simulator.run(transpiled, shots=n_shots).result()
    counts = result.get_counts()
    feature_vector = np.array([counts.get(f"{i:0{qc.num_qubits}b}", 0) / n_shots for i in range(2**qc.num_qubits)])
    return feature_vector

def quantum_feature_map(X, n_qubits):
    features = []
    for x in X:
        qc = build_quantum_reservoir(n_qubits, x)
        feat = extract_features(qc)
        features.append(feat)
    return np.array(features)

def train_quantum_classifier(X_train, y_train):
    clf = SVC(kernel='linear', class_weight='balanced')
    clf.fit(X_train, y_train)
    return clf
