import numpy as np
import torch
import torch.nn.functional as F

from saliency_bbox import saliency_bbox

from quanv import quanv
def QuanSaliencyMixReverse(batch, labels, beta=1.0, n_wires=4):
    lam = np.random.beta(beta, beta)
    device = batch.device
    B, C, H, W = batch.shape
    rand_index = torch.randperm(B, device=device)

    batch = batch.detach()
    #quantum_batch = quan_batch[rand_index].detach() # lần này là ảnh nguồn
    labels_a = labels.repeat_interleave(n_wires)
    labels_b = labels[rand_index].repeat_interleave(n_wires)


    source_batch = batch[rand_index] # chuyển qua quantum (32x32)
    #source_batch = F.interpolate(source_batch, scale_factor=0.5, mode='bilinear', align_corners=False) #(16x16)
    for _ in range(source_batch.shape[0]):
        quantum_masks = quanv(source_batch[_])
        source_batch[_] = F.interpolate(source_batch[_], 
                                        scale_factor=0.5, 
                                        mode='bilinear', 
                                        align_corners=False) * quantum_masks.unsqueeze(1)
    bbx1, bby1, bbx2, bby2 = saliency_bbox(source_batch[0], lam)

    processed_images = F.interpolate(batch, scale_factor=0.5, mode='bilinear', align_corners=False).repeat_interleave(n_wires, dim=0)
    #processed_images = img_resized * quantum_b.unsqueeze(1) 

    expanded_rand_index = rand_index.repeat_interleave(n_wires)
    processed_images[:, :, bbx1:bbx2, bby1:bby2] = source_batch[expanded_rand_index, :, bbx1:bbx2, bby1:bby2]

    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (H * W //4))
    return processed_images.to(dtype=torch.float32), labels_a, labels_b, lam

if __name__=="__main__":
    print ("QuanMixManual ran.")