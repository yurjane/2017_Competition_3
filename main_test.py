import math
import cv2
import torch
import datetime
import os

import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from skimage.color import rgb2gray, rgb2lab, lab2rgb
from PIL import Image
from numba import jit


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.midlevel_resnet = models.resnet18(num_classes=62)

    def forward(self, alphabet) -> torch.Tensor:
        target = self.midlevel_resnet(alphabet)
        return target


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(DEVICE)

alphabet2number = {
    '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
    'A': 10, 'B': 11, 'C': 12, 'D': 13, 'E': 14, 'F': 15, 'G': 16, 'H': 17, 'I': 18, 'J': 19,
    'K': 20, 'L': 21, 'M': 22, 'N': 23, 'O': 24, 'P': 25, 'Q': 26, 'R': 27, 'S': 28, 'T': 29,
    'U': 30, 'V': 31, 'W': 32, 'X': 33, 'Y': 34, 'Z': 35, 'a': 36, 'b': 37, 'c': 38, 'd': 39,
    'e': 40, 'f': 41, 'g': 42, 'h': 43, 'i': 44, 'j': 45, 'k': 46, 'l': 47, 'm': 48, 'n': 49,
    'o': 50, 'p': 51, 'q': 52, 'r': 53, 's': 54, 't': 55, 'u': 56, 'v': 57, 'w': 58, 'x': 59,
    'y': 60, 'z': 61
}

trans = transforms.ToTensor()

number2alphabet = dict(zip(alphabet2number.values(), alphabet2number.keys()))

resnet = models.resnet18()

D = Discriminator().to(DEVICE)

dirlist = os.listdir('./test')
dirlist = sorted(dirlist, key=lambda x:int(x[:-4]))
if os.path.exists(r'./module.pkl'):
    D.load_state_dict(torch.load(r'./new_module.pkl'))
    targets = list()
    for dir in dirlist:
        print(dir)
        image = Image.open(f'./test/{dir}').resize((224 * 5, 224), resample=Image.BILINEAR)
        image = trans(image).to(DEVICE)
        chars = list()
        for sub in range(5):
            imageArr = image[..., sub * 224:(sub + 1) * 224].view((1, 3, 224, 224))
            labels = list(D(imageArr)[0])
            target = labels.index(max(labels))
            chars.append(number2alphabet[int(target)])
        print(''.join(chars))
        targets.append(''.join(chars))
    data = pd.DataFrame({'y':targets})
    data.to_csv('./test.csv')
else:
    pass
