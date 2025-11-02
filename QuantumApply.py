import numpy as np
import torch
import torch.nn.functional as F
from saliency_bbox import saliency_bbox

def QuantumApply(batch, quan_batch, labels, n_wires=4):
    device = batch.device
    B, C, H, W = batch.shape
    labels = labels.repeat_interleave(n_wires)
    quantum_batch = quan_batch.detach().clone()
    maximum = quantum_batch.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
    minimum = quantum_batch.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]
    quantum_batch = (quantum_batch - minimum) / (maximum - minimum + 1e-8)

    img_resized = F.interpolate(batch.detach().clone(), scale_factor=0.5, mode='bilinear', align_corners=False)
    img_resized = img_resized.repeat_interleave(n_wires, dim=0)
    processed_images = img_resized * quantum_batch.unsqueeze(1) 

    return processed_images.to(dtype=torch.float32), labels

if __name__=="__main__":
    print ("QuanApp ran.")