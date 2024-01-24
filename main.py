from functools import partial
import pennylane as qml
import numpy
from pennylane import numpy as np
from copy import deepcopy
from tqdm import tqdm
from setup import *

from anatz import Ansatz
from feature_map import FeatureMap

from matplotlib import pyplot as plt
 


ansatz = Ansatz(qubit_number, depth)
feature_map = FeatureMap(qubit_number)

dev = qml.device('lightning.qubit', wires=qubit_number)

@qml.qnode(dev, diff_method="adjoint", interface='autograd', max_diff=2)
def circuit(x, theta):
    feature_map.get_map_advanced(x)
    ansatz.get_circ(theta)
    obs = qml.PauliZ(0)
    for j in range(qubit_number-1):
        obs = obs @ qml.PauliZ(j+1)
    return qml.expval(obs)

theta = np.array(np.ones(3*depth*qubit_number), requires_grad=True)
x = np.array([np.linspace(0., 0.9, space_size)], requires_grad=True).repeat(qubit_number, axis=0).T
adam = qml.AdamOptimizer(stepsize=0.01)

exp = np.exp(-x)
#exp = np.linspace(0., 1., space_size)
losses = []

best_loss = 1000.
best_theta = []

def cost(x, theta):
    loss = 0
    f_b = 0
    for e, i in enumerate(x):
        f = circuit(i, theta)
        if e == f_0_index:
            f_b = f_0 - f
        f += f_b
        df_x, _ = qml.grad(circuit)(i, theta)
        ode = np.abs(np.array(F(df_x, f, i)))
        ## Add Mse
        loss += 0.1*np.mean(ode ** 2)
        ## If x = 0 add regularization
        #if e == f_0_index:
        #    loss += 10*np.abs(f - f_0) ** 2
        ## Add regularization term
        f_diff = f_true(i)
        loss += (1 - np.tanh(ep/float(epoch)))*np.mean(np.mean(np.abs(f - f_diff) ** 2))
    loss /= space_size
    print(loss._value)
    losses.append(loss._value)

    return loss

index = 0
for ep in tqdm(range(epoch)):    
        _, theta = adam.step(cost, x, theta)
            
    
#print(theta)
out = []
grads = []
for i in x:

    out += [circuit(i, theta)]
    grad_x, _ = qml.grad(circuit)(i, theta)
    grads += [np.mean(grad_x)]
out = np.array(out)
out = out + f_0 - out[0]
grads = np.array(grads)
plt.plot(x, out, color='r')

plt.plot(x ,grads, color='g', linestyle='dashed')
#plt.plot(x, F(grads, out, np.mean(x)), linestyle='dashed')
plt.plot(x, f_true(x[:,0]), color='b')
plt.legend(("f(x)", "df(x)/dx", "ODE"))
plt.show()

plt.plot(losses)
plt.show()