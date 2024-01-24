from pennylane import numpy as np
# exp
F = lambda df_x, f, x: df_x + l * f

qubit_number = 1
depth = 4
epoch=250
f_0 = 1.
f_0_index = 0
space_size=20
l = 8
kappa = 0.1

f_true = lambda x: np.exp(-l*kappa*x)*np.cos(l*x)