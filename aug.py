import random 
import torch
from kornia import morphology as morph

def get_random_kernel():
    k = torch.rand(3,3).round()
    k[1,1] = 1
    return k

class Erosion:
    def __init__(self):
        self.fn = morph.erosion

    def __call__(self, img):
        kernel = get_random_kernel()
        return self.fn(img.unsqueeze(0), kernel)[0]

class Dilation:
    def __init__(self):
        self.fn = morph.dilation

    def __call__(self, img):
        kernel = get_random_kernel()
        return self.fn(img.unsqueeze(0), kernel)[0]

class RandomApply(torch.nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p
    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)
