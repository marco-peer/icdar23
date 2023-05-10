import torch
import torch.nn.init
import torchvision.transforms as transforms
import numpy as np
import math

import cv2
from PIL import Image

normalize = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


class PILTransforms:


    mean_image = None

    def get(self, transform):
        t = getattr(self, str(transform), None)
        assert t is not None, 'transform {} not known'.format(transform)
        return t()

    @staticmethod
    def train_transforms():
        transform = transforms.Compose([
            ResizeAndPad(),
            normalize
        ])
        return transform

    @staticmethod
    def val_transforms():
        transform = transforms.Compose([
            ResizeAndPad(),
            normalize
        ])
        return transform

    @staticmethod
    def test_transforms():
        transform = transforms.Compose([
            ResizeAndPad(),
            normalize

        ])
        return transform

class ResizeAndPad:

    def __init__(self, height=32, width=128, pad=False):
        self.height = height
        self.width = width
        self.pad = pad
    def __call__(self, img):

        if not self.pad:
            t1 = transforms.Resize((self.height, self.width))
            return t1(img)

        w_ = img.size[0] / self.width
        h_ = img.size[1] / self.height

        if w_ > h_:
            a = w_
        else:
            a = h_

        h, w = int(img.size[1] / a), int(img.size[0] / a)

        t1 = transforms.Resize((h,w))

        pad_w = (self.width - w)
        pad_h = (self.height - h)
        t2 = transforms.Pad((math.ceil(pad_w/2), math.ceil(pad_h/2), math.floor(pad_w/2), math.floor(pad_h/2)), fill=255)
        return t2(t1(img))
