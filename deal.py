import os
import math
import tool
import cv2
import torch

import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from skimage.color import rgb2gray, rgb2lab, lab2rgb
from PIL import Image
from numba import jit
from tool import imageTool

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")




def imageDecolor(img:np.array) -> np.ndarray:
    # small_img = np.array(Image.fromarray(img).resize((75, 15)))
    imgLab = rgb2lab(img)
    thetaList = np.arange(0, 1.1, 0.1)
    max_e = np.inf
    bestw = None
    for w1 in thetaList:
        for w2 in thetaList:
            if w1 + w2 > 1.0:
                break
            w3 = 1.0 - w1 - w2
            # print(w1, w2, w3)
            e = colorContrast(imgLab, img, (w1, w2, w3))
            # print(e)
            if e < max_e:
                max_e = e
                bestw = (w1, w2, w3)
    imgGray = np.empty(img.shape[:2])
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            imgGray[y, x] = bestw[0] * img[y, x, 0] + bestw[1] * img[y, x, 1] + bestw[2] * img[y, x, 2]
    # imgGray = 0.6 * imgLab[..., 0] + 0.4 * imgGray
    return imgGray

@jit(nopython=True)
def colorContrast(imgLab:np.ndarray, imgRgb:np.ndarray, w:tuple) -> float:
    size = imgRgb.shape
    imgGray = np.zeros(size[:2])
    aSum = 0.0
    for y in range(size[0]):
        for x in range(size[1]):
            rgbs = imgRgb[y, x]
            imgGray[y, x] = rgbs[0] * w[0] + rgbs[1] * w[1] + rgbs[2] * w[2]
    for a_y in range(size[0]):
        for a_x in range(size[1]):
            for b_y in range(size[0]):
                for b_x in range(size[1]):
                    deltaGray = abs(imgGray[a_y, a_x] - imgGray[b_y, b_x])
                    imgLab_a = imgLab[a_y, a_x]
                    imgLab_b = imgLab[b_y, b_x]
                    deltaLab = 0
                    for sub in range(3):
                        deltaLab += (imgLab_a[sub] - imgLab_b[sub]) ** 2
                    deltaLab = math.sqrt(deltaLab)
                    aSum += (deltaGray - deltaLab) ** 2
            # print(aSum)
    return aSum

@jit(nopython=True)
def normalDistribution(x, mu, sigma):
    # print(math.sqrt(2 * math.pi) * sigma)
    # print(-1 * (x - mu) ** 2 / (2 * sigma ** 2))
    return math.exp(-1 * (x - mu) ** 2 / (2 * sigma ** 2)) / (math.sqrt(2 * math.pi) * sigma)

@jit(nopython=True)
def norm(x:np.ndarray, sign:float):
    size = x.shape
    aSum = 0
    if sign == math.inf or sign == np.inf:
        return x.max()
    sign = int(sign)
    for sub in range(size[0]):
        aSum += abs(x[sub]) if sign == 1 else x[sub] ** sign
    return aSum if sign == 1 else math.pow(aSum, 1 / sign)


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
    else:
       pass


def remove_background(img:np.ndarray) -> np.ndarray:
    size = img.shape
    fmeans = np.array([img[0, 0], img[size[0] - 1, 0], img[0, size[1] - 1], img[size[0] - 1, size[1] - 1]]).mean(axis=0)
    pixArr = list()
    for y in range(size[0]):
        for x in range(size[1]):
            pixArr.append(img[y, x])
    pixArr = np.array(pixArr)
    ameans = pixArr.mean(axis=0)
    allDelta = np.sqrt(np.power(fmeans - ameans, 2).sum()) * 2
    for y in range(size[0]):
        for x in range(size[1]):
            delta = img[y, x] - fmeans
            dis = np.sqrt(np.power(delta, 2).sum())
            if dis < allDelta:
                img[y, x] = np.array([255, 255, 255])
    return img


class load_data(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        img_original = None
        if self.transform is not None:
            img_original = self.transform(img)
            img_original = np.asarray(img_original)
            img_original = remove_background(img_original)
            img_original = imageDecolor(img_original)
            grays = np.reshape(img_original, (900,)).mean()
            img_original[img_original > grays] = 1
            img_original[img_original <= grays] = 0
            img_original = torch.from_numpy(img_original).unsqueeze(0).float()
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img_original, target


class imageLabel(nn.Module):

    def __init__(self):
        super(imageLabel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 256, kernel_size=3, stride=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 32, kernel_size=3, stride=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 2, kernel_size=3, stride=1),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Sequential(
            nn.Linear(968, 484),
            nn.LeakyReLU(0.2),
            nn.Linear(484, 186),
            nn.LeakyReLU(0.2),
            nn.Linear(186, 62)
        )

    def forward(self, x:torch.tensor):
        y = self.conv(x)
        y = y.view(x.size(0), 968)
        y = self.fc(y)
        return y

# batch_size = 64
# max_epoch = 200
#
# transform = transforms.Compose(None)
#
# dataset = load_data("./chars/", transform=transform)
# print(dataset)
# dataset = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# for epoch in range(max_epoch):
#     for idx, (images, traget) in enumerate(dataset):
#         pass



if __name__ == '__main__':
    dirList = os.listdir("./train/0")
    charDict = dict()
    for dir in dirList:
        aImage = Image.open(f'./train/0/{dir}')
        print(f'./train/0/{dir}')
        imageArray = np.array(aImage)
        for sub in range(5):
            imageArr = imageArray[:, sub * 30:(sub + 1) * 30]
            char = int(ascii(ord(dir[sub])))
            if char not in charDict.keys():
                charDict[char] = 0
                if not os.path.exists(f'./char/{char:03d}'):
                    os.mkdir(f'./char/{char:03d}')
            charDict[char] += 1
            img = Image.fromarray(imageArr)
            img = img.resize((224, 224), resample=Image.BILINEAR)
            img.save(f'./char/{char:03d}/{charDict[char]:06d}.jpg')
