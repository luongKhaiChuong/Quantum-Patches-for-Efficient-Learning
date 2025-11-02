import numpy as np
import torch
import torch.nn.functional as F
from saliency_bbox import saliency_bbox

def QuanSaliencyMixReverse(batch, quan_batch, labels, beta=1.0, n_wires=4):
    lam = np.random.beta(beta, beta)
    device = batch.device
    B, C, H, W = batch.shape
    rand_index = torch.randperm(B, device=device)

    batch = batch.detach()
    quantum_batch = quan_batch[rand_index].detach() # lần này là ảnh nguồn
    labels_a = labels.repeat_interleave(n_wires)
    labels_b = labels[rand_index].repeat_interleave(n_wires)

    maximum = quantum_batch.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
    minimum = quantum_batch.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]
    quantum_b = (quantum_batch - minimum) / (maximum - minimum + 1e-8)

    source_batch = batch[rand_index] # chuyển qua quantum (32x32)
    source_batch = F.interpolate(source_batch, scale_factor=0.5, mode='bilinear', align_corners=False) #(16x16)
    source_batch = source_batch * quantum_b.unsqueeze(1)
    bbx1, bby1, bbx2, bby2 = saliency_bbox(source_batch[0], lam)

    processed_images = F.interpolate(batch, scale_factor=0.5, mode='bilinear', align_corners=False).repeat_interleave(n_wires, dim=0)
    #processed_images = img_resized * quantum_b.unsqueeze(1) 

    expanded_rand_index = rand_index.repeat_interleave(n_wires)
    processed_images[:, :, bbx1:bbx2, bby1:bby2] = source_batch[expanded_rand_index, :, bbx1:bbx2, bby1:bby2]

    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (H * W //4))
    return processed_images.to(dtype=torch.float32), labels_a, labels_b, lam

if __name__=="__main__":
    print ("QuanMixReverse ran.")