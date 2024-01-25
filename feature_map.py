import pennylane as qml
from pennylane import numpy as np

class FeatureMap:
    
    def __init__(self, qubit_number) -> None:
        self.qubit_number = qubit_number
        
    def get_map_regular(self, vars):
        """
        Product feature map
        """
        for i in range(self.qubit_number):
            qml.RY(np.arcsin(vars[i])/2.)
            
    def get_map_advanced(self, vars):
        """
        """
        for i in range(self.qubit_number):
            qml.RY((i+1)*np.arcsin(np.mod(vars[i]+1., 2.)-1.)/2, wires=i)
    
    def get_cust(self, vars):
        """
        """
        for i in range(self.qubit_number):
            qml.RY((i+1)*(np.arcsin(vars[i]))/2, wires=i)

    def get_cheby(self, vars):
        """
        """
        for i in range(self.qubit_number):
            qml.RY((i + 1) * (np.arccos(vars[i])), wires=i)
