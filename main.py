from functools import partial
import pennylane as qml
import numpy
from pennylane import numpy as np

from tqdm import tqdm
from setup import *

from anatz import Ansatz
from feature_map import FeatureMap

from matplotlib import pyplot as plt
 


ansatz = Ansatz(qubit_number, depth)
feature_map = FeatureMap(qubit_number)

dev = qml.device('lightning.qubit', wires=qubit_number)

@qml.qnode(dev, diff_method="parameter-shift", interface='autograd', max_diff=2)
def circuit(x, theta):
    feature_map.get_map_advanced(x)
    ansatz.get_circ(theta)
    obs = qml.PauliZ(0)
    for j in range(qubit_number-1):
        obs = obs @ qml.PauliZ(j+1)
    return qml.expval(obs)

theta = np.array(np.ones(3*depth*qubit_number), requires_grad=True)
x = np.array([numpy.linspace(0., 1., space_size)], requires_grad=True).repeat(qubit_number, axis=0).T
adam = qml.AdamOptimizer()

exp = np.exp(-x)
losses = []

def cost(x, theta):
    
    f = circuit(x, theta)
    
    
    df_x, _ = qml.grad(circuit)(x, theta) 
    loss = np.abs(F(df_x, f, x))
    ## Add Mse
    loss = np.mean(loss ** 2)
    ## If x = 0 add regularization
    if x[0] == 0:
        loss += 10*np.abs(f - f_0) ** 2
    ## Add regularization term
    loss += np.mean(np.abs(f - exp[index]) ** 2)
    print(loss)
    losses.append(loss)
    return loss

index = 0
for _ in tqdm(range(epoch)):
        for e, i in enumerate(x):
            index = e 
            _, theta = adam.step(cost, i, theta)
            
    
#print(theta)
out = []
grads = []
for i in x:
    out += [circuit(i, theta)]
    grad_x, _ = qml.grad(circuit)(i, theta)
    grads += [np.mean(grad_x)]
out = np.array(out)
grads = np.array(grads)
plt.plot(out)
plt.plot(grads)
plt.plot(F(grads, out, np.mean(x)), linestyle='dashed')

plt.show()

plt.plot(losses)
plt.show()