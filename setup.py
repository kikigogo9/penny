from pennylane import numpy as np
import pennylane as qml
from anatz import Ansatz
from feature_map import FeatureMap
# exp
F = lambda df_x, f, x: df_x + l * f * (kappa + np.tan(x))

qubit_number = 6
depth = 10
epoch=250
f_0 = 1.
f_0_index = 0
space_size=20
l = 20
kappa = 0.1

f_true = lambda x: np.exp(-l*kappa*x)*np.cos(l*x)

theta = np.array(np.ones(3*depth*qubit_number), requires_grad=True)
x = np.array([np.linspace(0., 0.9, space_size)], requires_grad=True).repeat(qubit_number, axis=0).T
adam = qml.AdamOptimizer(stepsize=0.01)
losses = []

ansatz = Ansatz(qubit_number, depth)
feature_map = FeatureMap(qubit_number)

dev = qml.device('lightning.qubit', wires=qubit_number)