import pennylane as qml
from pennylane import numpy as np
from tqdm import tqdm
from setup import *
from matplotlib import pyplot as plt

@qml.qnode(dev, diff_method="adjoint", interface='autograd', max_diff=3)
def circuit(x, theta):
    feature_map.get_cheby(x)
    ansatz.get_circ(theta)
    obs = qml.PauliZ(0)
    for j in range(qubit_number-1):
        obs = obs @ qml.PauliZ(j+1)
    return qml.expval(obs)

def cost(x, theta):
    loss = 0
    f_b = 0
    for e, i in enumerate(x):
        f = circuit(i, theta)
        if e == f_0_index:
            f_b = f_0 - f
        f += f_b
        df_x, _ = qml.grad(circuit)(i, theta)

        ode = np.mean(np.abs(np.array(F(df_x, f, i))))
        ## Add Mse
        loss += np.mean(ode ** 2)
        ## Add regularization term
        f_diff = f_true(i)
        loss += np.mean(np.mean(np.abs(f - f_diff) ** 2)) * (1 - np.tanh(max(ep - 50., 0)/float(epoch)))
    loss /= space_size
    print(loss._value)
    losses.append(loss._value)

    return loss

for ep in tqdm(range(epoch)):
        _, theta = adam.step(cost, x, theta)
        if ep + 1 % 100 == 0:
            adam.stepsize *= 0.1


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
plt.yscale('log')
plt.show()