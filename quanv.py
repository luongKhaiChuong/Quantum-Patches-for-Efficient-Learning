import warnings
warnings.filterwarnings("ignore")
import torch
import os
import numpy as np
import pennylane as qml
from pennylane.templates import RandomLayers
import torch.nn as nn
from torchvision import datasets, transforms
import time
from tqdm import tqdm

n_layers = 1
n_wires = 4 
channels = 3
if (pow(np.sqrt(n_wires), 2)!= n_wires):
    raise ValueError ("n_wires must be a square number")

dev = qml.device("default.qubit", wires=n_wires)
# Random circuit parameters
rand_params = np.random.uniform(high=2 * np.pi, size=(n_layers, n_wires))

@qml.qnode(dev)
def circuit(phi):
    # Encoding of 4 classical input values
    for j in range(n_wires):
        qml.RX(np.pi * phi[j], wires=j) #
        qml.RY(np.pi * phi[j], wires=j)
    # Random quantum circuit
    RandomLayers(rand_params, wires=list(range(n_wires)))

    # Measurement producing 4 classical output values
    return [qml.expval(qml.PauliZ(j)) for j in range(n_wires)]

def quanv(image, stride = 1):
    stride -= 1
    image_height, image_width = image.shape
    quan_kernel = int(np.sqrt(n_wires))
    if image_height % quan_kernel != 0 or image_width % quan_kernel != 0:
        raise ValueError(f"Image dimensions must be divisible by {quan_kernel}")

    out_height = image_height // (quan_kernel + stride)
    out_width = image_width // (quan_kernel + stride)
    
    out = np.zeros((n_wires, out_height, out_width))
    unfold = nn.Unfold(kernel_size=quan_kernel, stride=quan_kernel)
    input_tensor = unfold(image.unsqueeze(0)).squeeze(0)
    quantumed_tensor = circuit(input_tensor)
    quantumed_tensor = torch.tensor(np.array(quantumed_tensor), dtype=torch.float32)
    output_tensor = quantumed_tensor.resize(n_wires, out_height, out_width)

    return output_tensor

if __name__ == "__main__":
    print ("Quanv ran.")