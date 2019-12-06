import os

import math
import cv2
import torch

import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from skimage.color import rgb2gray, rgb2lab, lab2rgb
from PIL import Image
from numba import jit
import datetime

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

number2alphabet = dict(zip(alphabet2number.values(), alphabet2number.keys()))

class get_verification_code(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        img_original: torch.Tensor = None
        if self.transform is not None:
            img_original = self.transform(img).to(DEVICE)
            # img_original = F.interpolate(img_original, size=size, mode='bilinear')
        if self.target_transform is not None:
            target = self.target_transform(target)
        # targets = torch.zeros(62).to(DEVICE)
        # targets[target] = 1
        return img_original, target


resnet = models.resnet18(num_classes=62)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.midlevel_resnet = models.resnet18(num_classes=62)

    def forward(self, alphabet) -> torch.Tensor:
        target = self.midlevel_resnet(alphabet)
        return target


max_epoch = 100
batch_size = 64

dataset = get_verification_code('./char', transform=transforms.ToTensor())
dataset = DataLoader(dataset, batch_size=batch_size, shuffle=True)

D = Discriminator().to(DEVICE)
if os.path.exists(r'./module.pkl'):
    D.load_state_dict(torch.load(r'./module.pkl'))
    print(True)
else:
    print(False)
    pass

criterion = nn.CrossEntropyLoss()

D_opt = torch.optim.Adam(D.parameters(), lr=0.02, betas=(0.5, 0.999))

step = 0
for epoch in range(max_epoch):
    for idx, (img, target) in enumerate(dataset):
        target = target.to(DEVICE)
        pred = D(img)
        loss = criterion(pred, target)
        D.zero_grad()
        loss.backward()
        D_opt.step()
        if step % 50 == 0:
            print('Epoch: {}/{}, Step: {}, D Loss: {}'.format(epoch, max_epoch, step, loss.item()))
        if step % 5 == 0:
            torch.save(D.state_dict(), r'./module.pkl')
            cur = datetime.datetime.now()
            print(f'now:{cur:%Y-%m-%d (%a) %H:%M:%S} save the modul')
        step += 1




