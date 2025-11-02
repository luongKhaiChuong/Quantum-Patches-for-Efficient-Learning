import numpy as np
import torch
from saliency_bbox import saliency_bbox
def SaliencyMix(batch, labels, beta=1.0):
    lam = np.random.beta(beta, beta)
    new_batch = batch.clone().detach()
    rand_index = torch.randperm(new_batch.size()[0])
    labels_a = labels
    labels_b = labels[rand_index]
    bbx1, bby1, bbx2, bby2 = saliency_bbox(new_batch[rand_index[0]], lam)
    new_batch[:, :, bbx1:bbx2, bby1:bby2] = new_batch[rand_index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (new_batch.size()[-1] * new_batch.size()[-2]))
    return new_batch, labels_a, labels_b, lam

if __name__=="__main__":
    print ("SaliencyMix ran.")