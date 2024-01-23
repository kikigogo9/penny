import pennylane as qml

class Ansatz:
    def __init__(self, qubit_number, depth) -> None:
        self.qubit_number = qubit_number
        self.depth = depth
        self    
    
    def get_circ(self, vars):
        for i in range(self.depth):
            for j in range(self.qubit_number):
                qml.RZ(vars[3 * i * self.qubit_number + 3* j + 0], wires=j)
                qml.RX(vars[3 * i * self.qubit_number + 3*j + 1], wires=j)
                qml.RZ(vars[3 * i * self.qubit_number + 3*j + 2], wires=j)
            for j in range(self.qubit_number - 1):
                if j % 2 == 0:
                    qml.CNOT(wires=[j, j+1])
            for j in range(self.qubit_number - 1):
                if j % 2 == 1:
                    qml.CNOT(wires=[j, j+1])